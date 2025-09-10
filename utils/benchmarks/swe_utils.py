import re
import os
import platform
from pathlib import Path
from typing import List, Dict, Any, Union

from schema import ItemEval
from utils.core_utils import get_logger
from swebench.harness.run_evaluation import main as run_swebench_evaluation

logger = get_logger(__name__)


_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Unfenced diff starting with 'diff --git' … until next blank-blank separation or EOF
_UNFENCED_DIFF_GIT_RE = re.compile(r"(?:^|\n)(diff --git[^\n]*\n(?:.*\n)+?)(?=\n(?=\S)|\Z)", re.DOTALL)
# Unfenced unified diff: requires --- / +++ headers and at least one @@ hunk
# Improved pattern that stops at the end of the actual diff content
_UNFENCED_UNIFIED_RE = re.compile(
    r"(?:^|\n)(?P<body>(?:Index:[^\n]*\n)?(?:---[^\n]*\n\+\+\+[^\n]*\n)(?:@@[^\n]*\n(?:[^@\n][^\n]*\n)*)+?)(?=\n\n|\n(?=[A-Za-z])|\Z)",
    re.DOTALL | re.MULTILINE
)

def _has_clean_diff_structure(text: str) -> bool:
    """
    Check if the text has a clean diff structure without explanatory text mixed in.
    Uses structural analysis to detect when content stops being diff-like.
    """
    lines = text.split('\n')
    
    # Find where the actual diff content starts
    diff_start = -1
    for i, line in enumerate(lines):
        if line.startswith('---') and i + 1 < len(lines) and lines[i + 1].startswith('+++'):
            diff_start = i
            break
        elif line.startswith('diff --git'):
            diff_start = i
            break
        elif line.startswith('Index:'):
            diff_start = i
            break
    
    if diff_start == -1:
        return False
    
    # Analyze the structure from the diff start
    in_hunk = False
    consecutive_non_diff_lines = 0
    
    for i in range(diff_start, len(lines)):
        line = lines[i].rstrip()
        
        if (line.startswith('---') or line.startswith('+++') or 
            line.startswith('diff --git') or line.startswith('Index:') or
            line.startswith('===')):
            # Diff headers are valid
            consecutive_non_diff_lines = 0
            
        elif line.startswith('@@'):
            # Hunk header
            in_hunk = True
            consecutive_non_diff_lines = 0
            
        elif in_hunk and (line.startswith(' ') or line.startswith('+') or line.startswith('-')):
            # Valid diff content within hunk
            consecutive_non_diff_lines = 0
            
        elif line.strip() == '':
            # Empty lines can be part of diff context, but track them
            consecutive_non_diff_lines += 0.5  # Count empty lines as half
            
        else:
            # Line that doesn't match diff format
            consecutive_non_diff_lines += 1
            
            # If we have any significant non-diff content after the actual diff, it's not clean
            # Allow for 1 trailing empty line, but any actual content means it's mixed
            if consecutive_non_diff_lines >= 1:
                return False
    
    return True

def _looks_like_unified_diff(text: str) -> bool:
    t = text.strip()
    if t.startswith("diff --git"):
        return True
    has_headers = re.search(r"^---[^\n]*\n\+\+\+[^\n]*", t, re.MULTILINE) is not None
    has_hunk = re.search(r"^@@", t, re.MULTILINE) is not None
    
    # Also check for fenced code blocks - if it contains ``` it's likely mixed content
    has_fenced_blocks = "```" in t
    
    # Check if this has proper diff structure throughout (no explanatory text mixed in)
    has_clean_structure = _has_clean_diff_structure(t)
    
    # For a text to be considered a pure diff, it should:
    # 1. Have diff headers and hunks
    # 2. NOT have fenced code blocks  
    # 3. Start with --- or diff --git or Index: (be properly formatted)
    # 4. Have clean diff structure throughout (no mixed explanatory text)
    starts_properly = t.startswith("---") or t.startswith("diff --git") or t.startswith("Index:")
    
    return has_headers and has_hunk and not has_fenced_blocks and starts_properly and has_clean_structure

def _clean_fence(text: str) -> str:
    # Remove any leading "patch"/"diff" hints accidentally copied inside the fence
    return text.strip()

def _clean_extracted_patch(text: str) -> str:
    """
    Clean up extracted patch text by removing explanatory content before/after the actual patch.
    Uses structural analysis rather than word-based detection.
    """
    lines = text.split('\n')
    
    # Find the start of the actual patch (--- line)
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('---') and i + 1 < len(lines) and lines[i + 1].startswith('+++'):
            start_idx = i
            break
    
    if start_idx == -1:
        return text  # No proper patch headers found
    
    # Find the end of the patch content using structural analysis
    end_idx = len(lines)
    last_valid_patch_line = start_idx + 1  # At least include the +++ line
    
    # Track state through the diff
    expecting_hunk = True
    in_hunk = False
    consecutive_non_patch_lines = 0
    
    for i in range(start_idx + 2, len(lines)):  # Start after ---/+++ headers
        line = lines[i].rstrip()
        
        if line.startswith('@@'):
            # Hunk header - this is valid patch content
            in_hunk = True
            expecting_hunk = False
            last_valid_patch_line = i
            consecutive_non_patch_lines = 0
            
        elif in_hunk and (line.startswith(' ') or line.startswith('+') or line.startswith('-')):
            # Valid diff line within a hunk
            last_valid_patch_line = i
            consecutive_non_patch_lines = 0
            
        elif line.strip() == '':
            # Empty line - could be part of diff context or separation
            # Don't count empty lines as "non-patch" immediately
            if in_hunk:
                # Empty line within a hunk can be valid context
                last_valid_patch_line = i
                consecutive_non_patch_lines = 0
            else:
                consecutive_non_patch_lines += 1
                
        elif expecting_hunk and not line.startswith('@@'):
            # We're expecting a hunk but got something else - likely explanatory text
            end_idx = last_valid_patch_line + 1
            break
            
        elif in_hunk and not line.startswith('@@') and not line.startswith(' ') and not line.startswith('+') and not line.startswith('-'):
            # We're in a hunk but this line doesn't follow diff format
            # This is likely where explanatory text begins
            consecutive_non_patch_lines += 1
            
            # If we have multiple consecutive lines that don't look like diff content,
            # it's probably explanatory text
            if consecutive_non_patch_lines >= 2:
                end_idx = last_valid_patch_line + 1
                break
                
        else:
            consecutive_non_patch_lines += 1
            # If we see too many non-patch lines, stop
            if consecutive_non_patch_lines >= 2:
                end_idx = last_valid_patch_line + 1
                break
    
    # Ensure we include at least the headers
    end_idx = max(end_idx, last_valid_patch_line + 1)
    
    result = '\n'.join(lines[start_idx:end_idx]).strip()
    return result

def answer_to_patch(answer: str) -> str:
    """
    Extract a unified diff patch from a model answer for SWE-bench.
    Returns a string suitable for `git apply`. Empty string means "no patch".
    """
    if not answer or not answer.strip():
        return ""

    # Fast path: if the whole answer already looks like a diff, return as-is (after LF normalization).
    whole = answer.strip()
    if _looks_like_unified_diff(whole):
        return whole.replace("\r\n", "\n").rstrip() + "\n"

    patches = []

    # 1) Diff-fenced code blocks: ```diff ...``` or ```patch ...```
    for m in _DIFF_FENCE_RE.finditer(answer):
        block = _clean_fence(m.group(1))
        if _looks_like_unified_diff(block):
            patches.append(block)

    # Only look for unfenced diffs if we didn't find any fenced ones
    # This prevents duplicates when content appears both fenced and unfenced
    if not patches:
        # 2) Unfenced 'diff --git' blocks
        for m in _UNFENCED_DIFF_GIT_RE.finditer(answer):
            block = m.group(1).strip()
            if _looks_like_unified_diff(block):
                patches.append(block)

        # 3) Unfenced unified diff blocks with ---/+++ and hunks
        #    (helps when models omit the 'diff --git' headers)
        for m in _UNFENCED_UNIFIED_RE.finditer(answer):
            block = m.group("body").strip()
            # Additional filtering to remove obvious non-patch content
            clean_block = _clean_extracted_patch(block)
            if clean_block and _looks_like_unified_diff(clean_block):
                patches.append(clean_block)

    if not patches:
        # Nothing that `git apply` would accept
        return ""

    # Normalize newlines, ensure trailing newline, and separate multiple diffs with one blank line
    normalized = "\n\n".join(p.replace("\r\n", "\n").strip() for p in patches).rstrip() + "\n"
    return normalized



def prepare_swebench_predictions(
    predictions: List[ItemEval], 
    model_name: str
) -> List[Dict[str, Any]]:
    """
    Convert evaluation results to SWE-bench prediction format.
    
    Args:
        predictions: Predictions from the model
        model_name: Name of the model being evaluated
        
    Returns:
        List of output in SWE-bench format
    """
    swe_bench_predictions = []

    for pred in predictions:
        if pred.decision == "answer" and pred.answer.strip():
            patch = answer_to_patch(pred.answer)
            swe_bench_predictions.append({
                "instance_id": pred.evaluation_metadata["instance_id"],
                "model_patch": patch,
                "model_name_or_path": model_name
            })
    
    return swe_bench_predictions


def run_swebench_with_docker(
    predictions_path: str,
    dataset_name: str,
    split: str,
    run_id: str,

) -> Union[Path, None]:
    """
    Run SWE-bench evaluation using the official harness.

    Args:
        predictions_path: Path to the predictions file
        dataset_name: Name of the dataset
        split: Split of the dataset
        run_id: ID of the run
    
    Returns:
        SWE report file path. Returns None if evaluation fails.
    """
    # Fix Docker socket path for macOS Docker Desktop
    if platform.system() == "Darwin" and not os.environ.get("DOCKER_HOST"):
        docker_sock_path = os.path.expanduser("~/.docker/run/docker.sock")
        if os.path.exists(docker_sock_path):
            os.environ["DOCKER_HOST"] = f"unix://{docker_sock_path}"
            logger.info(f"Set DOCKER_HOST to {os.environ['DOCKER_HOST']} for macOS Docker Desktop")
    
    # Run the official SWE-bench evaluation
    try:
        swebench_report_path = run_swebench_evaluation(
            dataset_name=dataset_name, 
            split=split, 
            predictions_path=predictions_path, 
            run_id=run_id, 
                # default values from swebench.harness.run_evaluation
            max_workers=int(os.cpu_count() * 0.75),
            force_rebuild=False,
            cache_level="env", 
            clean=False,
            open_file_limit=4096,
            timeout=1_800, 
            namespace="swebench", 
            rewrite_reports=False,
            modal=False,
            instance_image_tag="latest", 
            report_dir=".",
            instance_ids=[],
        )
        logger.info(f"SWE-bench evaluation completed. Results in evaluation_results/{run_id}/")
        return swebench_report_path
    except Exception as e:
        logger.error(f"SWE-bench evaluation failed: {e}")
        return None


