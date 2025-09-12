import re

from utils.benchmarks.swe_utils import answer_to_patch


def validate_diff_structure(diff_text: str) -> dict:
    """
    Comprehensive structural validation of diff content.
    Returns a dict with validation results and structural information.
    """
    if not diff_text or not diff_text.strip():
        return {"valid": False, "error": "Empty diff"}

    lines = diff_text.split("\n")
    result = {
        "valid": True,
        "error": None,
        "headers": [],
        "hunks": [],
        "total_lines": len(lines),
        "added_lines": 0,
        "removed_lines": 0,
        "context_lines": 0,
        "has_git_header": False,
        "has_index_header": False,
        "ends_with_newline": diff_text.endswith("\n"),
        "starts_properly": False,
        "has_explanatory_text": False,
    }

    i = 0
    # Check for git diff header
    if i < len(lines) and lines[i].startswith("diff --git"):
        result["has_git_header"] = True
        result["starts_properly"] = True
        i += 1
        # Skip optional lines (index, mode changes, etc.)
        while i < len(lines) and not lines[i].startswith("---"):
            if lines[i].startswith("index"):
                result["has_index_header"] = True
            i += 1

    # Check for Index header
    if i < len(lines) and lines[i].startswith("Index:"):
        result["has_index_header"] = True
        result["starts_properly"] = True
        # Skip ===== separator
        i += 1
        if i < len(lines) and lines[i].startswith("==="):
            i += 1

    # Look for --- header
    if i < len(lines) and lines[i].startswith("---"):
        result["starts_properly"] = True
        result["headers"].append(("from", lines[i]))
        i += 1

        # Must be followed by +++ header
        if i < len(lines) and lines[i].startswith("+++"):
            result["headers"].append(("to", lines[i]))
            i += 1
        else:
            result["valid"] = False
            result["error"] = "Missing +++ header after ---"
            return result
    else:
        result["valid"] = False
        result["error"] = "Missing --- header"
        return result

    # Parse hunks
    current_hunk = None
    in_hunk = False

    while i < len(lines):
        line = lines[i]

        if line.startswith("@@"):
            # New hunk header
            if not re.match(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", line):
                result["valid"] = False
                result["error"] = f"Invalid hunk header: {line}"
                return result

            if current_hunk:
                result["hunks"].append(current_hunk)

            current_hunk = {
                "header": line,
                "lines": [],
                "context": 0,
                "added": 0,
                "removed": 0,
            }
            in_hunk = True

        elif in_hunk and line.startswith(" "):
            # Context line
            current_hunk["lines"].append(("context", line))
            current_hunk["context"] += 1
            result["context_lines"] += 1

        elif in_hunk and line.startswith("+"):
            # Added line
            current_hunk["lines"].append(("added", line))
            current_hunk["added"] += 1
            result["added_lines"] += 1

        elif in_hunk and line.startswith("-"):
            # Removed line
            current_hunk["lines"].append(("removed", line))
            current_hunk["removed"] += 1
            result["removed_lines"] += 1

        elif line.strip() == "":
            # Empty line - could be end of hunk or just spacing
            if current_hunk:
                current_hunk["lines"].append(("empty", line))

        else:
            # Non-diff content after diff - likely explanatory text
            if in_hunk or current_hunk:
                result["has_explanatory_text"] = True
                # Don't mark as invalid, but note it
                break

        i += 1

    # Add the last hunk
    if current_hunk:
        result["hunks"].append(current_hunk)

    # Validation checks
    if not result["hunks"]:
        result["valid"] = False
        result["error"] = "No valid hunks found"

    return result


class TestAnswerToPatch:
    """Test cases for the answer_to_patch function."""

    def test_empty_input(self):
        """Test with empty or None input."""
        assert answer_to_patch("") == ""
        assert answer_to_patch(None) == ""
        assert answer_to_patch("   ") == ""

    def test_valid_unified_diff(self):
        """Test with a valid unified diff format."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("world")
+    print("universe")
 return True"""

        expected = diff + "\n"
        result = answer_to_patch(diff)
        assert result == expected

        # Comprehensive structural validation
        validation = validate_diff_structure(result)
        assert validation["valid"], f"Diff structure invalid: {validation['error']}"
        assert validation["starts_properly"]
        assert validation["ends_with_newline"]
        assert len(validation["headers"]) == 2
        assert validation["headers"][0][1] == "--- a/file.py"
        assert validation["headers"][1][1] == "+++ b/file.py"
        assert len(validation["hunks"]) == 1
        assert validation["hunks"][0]["header"] == "@@ -1,3 +1,3 @@"
        assert validation["added_lines"] == 1
        assert validation["removed_lines"] == 1
        assert validation["context_lines"] == 2
        assert not validation["has_explanatory_text"]

    def test_empty_diff(self):
        """Test with empty diff."""
        diff = ""
        result = answer_to_patch(diff)
        assert result == ""

    def test_git_diff_format(self):
        """Test with git diff format."""
        diff = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -10,7 +10,7 @@ class TestClass:
     def method(self):
-        old_code()
+        new_code()
         return result"""

        result = answer_to_patch(diff)
        assert result.startswith("diff --git")
        assert "--- a/test.py" in result
        assert "+++ b/test.py" in result
        assert result.endswith("\n")

        # Comprehensive structural validation
        validation = validate_diff_structure(result)
        assert validation["valid"], f"Git diff structure invalid: {validation['error']}"
        assert validation["has_git_header"]
        assert validation["has_index_header"]
        assert validation["starts_properly"]
        assert validation["ends_with_newline"]
        assert len(validation["headers"]) == 2
        assert validation["headers"][0][1] == "--- a/test.py"
        assert validation["headers"][1][1] == "+++ b/test.py"
        assert len(validation["hunks"]) == 1
        assert "@@ -10,7 +10,7 @@" in validation["hunks"][0]["header"]
        assert validation["added_lines"] == 1
        assert validation["removed_lines"] == 1
        assert validation["context_lines"] == 2
        assert not validation["has_explanatory_text"]

    def test_fenced_diff_block(self):
        """Test with fenced code block containing diff."""
        answer = """Here's the fix:

```diff
--- a/example.py
+++ b/example.py
@@ -5,7 +5,7 @@ def function():
     if condition:
-        old_behavior()
+        new_behavior()
     return value
```

This should resolve the issue."""

        result = answer_to_patch(answer)
        assert result.startswith("--- a/example.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/example.py"
        assert lines[1] == "+++ b/example.py"
        assert "old_behavior()" in result
        assert "new_behavior()" in result
        assert result.endswith("\n")

        # Comprehensive structural validation
        validation = validate_diff_structure(result)
        assert validation["valid"], (
            f"Fenced diff structure invalid: {validation['error']}"
        )
        assert validation["starts_properly"]
        assert validation["ends_with_newline"]
        assert not validation["has_git_header"]  # Extracted from fenced block
        assert len(validation["headers"]) == 2
        assert validation["headers"][0][1] == "--- a/example.py"
        assert validation["headers"][1][1] == "+++ b/example.py"
        assert len(validation["hunks"]) == 1
        assert "@@ -5,7 +5,7 @@" in validation["hunks"][0]["header"]
        assert validation["added_lines"] == 1
        assert validation["removed_lines"] == 1
        assert validation["context_lines"] == 2
        assert not validation["has_explanatory_text"], (
            "Should not contain explanatory text from outside the fence"
        )

    def test_fenced_patch_block(self):
        """Test with fenced patch block."""
        answer = """```patch
--- a/module.py
+++ b/module.py
@@ -1,4 +1,4 @@
 def calculate():
-    return x * 2
+    return x * 3
 # end function
```"""

        result = answer_to_patch(answer)
        assert result.startswith("--- a/module.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/module.py"
        assert lines[1] == "+++ b/module.py"
        assert "return x * 3" in result
        assert result.endswith("\n")

    def test_multiple_diff_blocks(self):
        """Test with multiple diff blocks in the answer."""
        answer = """First fix:
```diff
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-old_line1
+new_line1
```

Second fix:
```diff
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-old_line2
+new_line2
```"""

        result = answer_to_patch(answer)
        # Should start with the first diff
        assert result.startswith("--- a/file1.py")

        # Split by double newlines to get individual diffs
        diffs = result.strip().split("\n\n")
        assert len(diffs) >= 2  # Should have at least 2 diff blocks

        # First diff should start properly
        first_diff_lines = diffs[0].split("\n")
        assert first_diff_lines[0] == "--- a/file1.py"
        assert first_diff_lines[1] == "+++ b/file1.py"

        assert "new_line1" in result
        assert "new_line2" in result
        assert "--- a/file2.py" in result
        assert "Second fix:" not in result

    def test_unfenced_git_diff(self):
        """Test with unfenced git diff in the answer."""
        answer = """The issue can be fixed with this change:

diff --git a/src/utils.py b/src/utils.py
index abcd123..efgh456 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -15,6 +15,8 @@ def process_data(data):
     if not data:
         return None
+
+    # Added validation
     return process(data)

This ensures proper validation."""

        result = answer_to_patch(answer)
        # The function extracts the git diff portion and strips explanatory text
        assert "diff --git a/src/utils.py b/src/utils.py" in result
        assert "Added validation" in result
        assert result.endswith("\n")

    def test_unfenced_unified_diff(self):
        """Test with unfenced unified diff (no git headers)."""
        answer = """Here's the fix:

--- a/config.py
+++ b/config.py
@@ -10,5 +10,5 @@ SETTINGS = {
-    'debug': True,
+    'debug': False,
 }

This disables debug mode."""

        result = answer_to_patch(answer)
        assert result.startswith("--- a/config.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/config.py"
        assert lines[1] == "+++ b/config.py"
        assert "'debug': False" in result

    def test_explanatory_text_filtering(self):
        """Test that explanatory text is filtered out from patches."""
        answer = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def test():
-    old_code()
+    new_code()

However, note that this change might have side effects.
You should also consider updating the documentation.
The above patch modifies the core functionality."""

        result = answer_to_patch(answer)
        # Should start with proper diff headers
        assert result.startswith("--- a/test.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/test.py"
        assert lines[1] == "+++ b/test.py"
        assert lines[2].startswith("@@")

        # Should not contain explanatory text
        assert "However" not in result
        assert "You should" not in result
        assert "The above patch" not in result
        # Should contain the actual diff
        assert "new_code()" in result
        assert result.endswith("\n")

        # Comprehensive structural validation - should be clean after filtering
        validation = validate_diff_structure(result)
        assert validation["valid"], (
            f"Filtered diff structure invalid: {validation['error']}"
        )
        assert validation["starts_properly"]
        assert validation["ends_with_newline"]
        assert len(validation["headers"]) == 2
        assert validation["headers"][0][1] == "--- a/test.py"
        assert validation["headers"][1][1] == "+++ b/test.py"
        assert len(validation["hunks"]) == 1
        assert validation["hunks"][0]["header"] == "@@ -1,3 +1,3 @@"
        assert validation["added_lines"] == 1
        assert validation["removed_lines"] == 1
        assert validation["context_lines"] == 1
        assert not validation["has_explanatory_text"], (
            "Explanatory text should be filtered out"
        )

        # Verify that the original input WOULD have explanatory text
        original_validation = validate_diff_structure(answer)
        assert original_validation["has_explanatory_text"], (
            "Original should have had explanatory text"
        )

    def test_no_valid_diff_content(self):
        """Test with answer containing no valid diff content."""
        answer = """The issue is in the function. You need to change the logic.

The problem occurs because the variable is not initialized properly.
I recommend updating the code to handle edge cases better."""

        result = answer_to_patch(answer)
        assert result == ""

    def test_malformed_diff(self):
        """Test with malformed diff that shouldn't be accepted."""
        answer = """```diff
This is not a proper diff
just some random text
```"""

        result = answer_to_patch(answer)
        assert result == ""

    def test_carriage_return_normalization(self):
        """Test that carriage returns are normalized."""
        diff = "--- a/file.py\r\n+++ b/file.py\r\n@@ -1,1 +1,1 @@\r\n-old\r\n+new"

        result = answer_to_patch(diff)
        assert "\r\n" not in result
        assert "\n" in result
        assert result.endswith("\n")

    def test_case_insensitive_fenced_blocks(self):
        """Test that fenced blocks work with different cases."""
        answer = """```DIFF
--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
```"""

        result = answer_to_patch(answer)
        assert "--- a/test.py" in result
        assert "+new" in result

    def test_patch_with_index_header(self):
        """Test patch with Index: header."""
        answer = """Index: src/main.py
===================================================================
--- a/src/main.py
+++ b/src/main.py
@@ -5,7 +5,7 @@ def main():
     print("Starting application")
-    old_function()
+    new_function()
     print("Done")"""

        result = answer_to_patch(answer)
        assert "Index: src/main.py" in result
        assert "new_function()" in result

    def test_mixed_content_with_diff(self):
        """Test answer with mixed explanatory content and valid diff."""
        answer = """The bug is in the validation function. Here's the issue:

The current implementation doesn't handle empty strings correctly.

```diff
--- a/validator.py
+++ b/validator.py
@@ -8,6 +8,8 @@ def validate_input(value):
     if value is None:
         return False
+    if value == "":
+        return False
     return len(value) > 0
```

This change ensures empty strings are properly rejected."""

        result = answer_to_patch(answer)
        assert result.startswith("--- a/validator.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/validator.py"
        assert lines[1] == "+++ b/validator.py"
        assert 'if value == ""' in result
        # The function should extract only the diff from fenced blocks
        assert result.count("--- a/validator.py") == 1  # Should appear only once

    def test_complex_diff_with_context(self):
        """Test complex diff with more context lines."""
        diff = """--- a/complex.py
+++ b/complex.py
@@ -15,12 +15,15 @@ class DataProcessor:
         self.config = config
         self.cache = {}

     def process(self, data):
+        if not self.validate_data(data):
+            raise ValueError("Invalid data")
+
         if data in self.cache:
             return self.cache[data]

         result = self._expensive_operation(data)
-        self.cache[data] = result
+        self.cache[data] = self._sanitize_result(result)
         return result"""

        result = answer_to_patch(diff)
        # Should start with proper diff headers
        assert result.startswith("--- a/complex.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/complex.py"
        assert lines[1] == "+++ b/complex.py"
        assert lines[2].startswith("@@")

        # Should contain the expected changes
        assert "class DataProcessor:" in result
        assert "validate_data(data)" in result
        assert "_sanitize_result(result)" in result
        assert result.endswith("\n")

    def test_whitespace_handling(self):
        """Test proper handling of whitespace in diffs."""
        answer = """

```diff
--- a/whitespace.py
+++ b/whitespace.py
@@ -1,3 +1,3 @@
 def func():
-    return True
+    return False
```

   """

        result = answer_to_patch(answer)
        # The function should extract the diff content from the fenced block
        assert result.startswith("--- a/whitespace.py")
        lines = result.split("\n")
        assert lines[0] == "--- a/whitespace.py"
        assert lines[1] == "+++ b/whitespace.py"
        assert "+    return False" in result
        assert result.endswith("\n")


class TestStructuralIntegrity:
    """Additional tests focused on structural integrity and edge cases."""

    def test_multiple_hunks_validation(self):
        """Test diff with multiple hunks has correct structure."""
        diff = """--- a/complex.py
+++ b/complex.py
@@ -10,7 +10,7 @@ class Example:
     def method1(self):
-        old_implementation()
+        new_implementation()
         return result

@@ -25,6 +25,8 @@ class Example:
     def method2(self):
         existing_code()
+        # Added comment
+        added_functionality()
         return value"""

        result = answer_to_patch(diff)
        validation = validate_diff_structure(result)

        assert validation["valid"], f"Multi-hunk diff invalid: {validation['error']}"
        assert len(validation["hunks"]) == 2, "Should have exactly 2 hunks"

        # First hunk
        hunk1 = validation["hunks"][0]
        assert "@@ -10,7 +10,7 @@" in hunk1["header"]
        assert hunk1["added"] == 1
        assert hunk1["removed"] == 1
        assert hunk1["context"] == 2

        # Second hunk
        hunk2 = validation["hunks"][1]
        assert "@@ -25,6 +25,8 @@" in hunk2["header"]
        assert hunk2["added"] == 2
        assert hunk2["removed"] == 0
        assert hunk2["context"] == 3

        # Total counts
        assert validation["added_lines"] == 3
        assert validation["removed_lines"] == 1
        assert validation["context_lines"] == 5

    def test_malformed_hunk_header_detection(self):
        """Test that malformed hunk headers are detected."""
        bad_diff = """--- a/test.py
+++ b/test.py
@@ invalid hunk header @@
-old line
+new line"""

        result = answer_to_patch(bad_diff)

        # Our validation should catch this as invalid
        validation = validate_diff_structure(result)
        assert not validation["valid"], (
            "Malformed hunk header should be detected as invalid"
        )
        assert "Invalid hunk header" in validation["error"]

        # Test another type of malformed header - this one is actually valid format
        # @@ -1 +1 @@ is equivalent to @@ -1,1 +1,1 @@ so let's test a truly invalid one
        bad_diff2 = """--- a/test.py
+++ b/test.py
@@ not-a-number,1 +1,1 @@
-old
+new"""

        result2 = answer_to_patch(bad_diff2)
        validation2 = validate_diff_structure(result2)
        if result2:  # If function doesn't reject it outright
            assert not validation2["valid"], (
                "Truly malformed hunk header should be detected"
            )

    def test_mixed_line_endings(self):
        """Test handling of mixed line endings."""
        diff_with_crlf = (
            "--- a/test.py\r\n+++ b/test.py\r\n@@ -1,1 +1,1 @@\r\n-old\r\n+new"
        )

        result = answer_to_patch(diff_with_crlf)
        validation = validate_diff_structure(result)

        assert validation["valid"], f"CRLF diff invalid: {validation['error']}"
        assert "\r\n" not in result, "Should normalize CRLF to LF"
        assert validation["ends_with_newline"]

    def test_empty_hunks_detection(self):
        """Test detection of diffs without actual content changes."""
        empty_diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 line1
 line2
 line3"""

        result = answer_to_patch(empty_diff)
        if result:  # Function might return this as-is or reject it
            validation = validate_diff_structure(result)
            assert validation["valid"]
            # This diff has no actual changes (all context)
            assert validation["added_lines"] == 0
            assert validation["removed_lines"] == 0
            assert validation["context_lines"] == 3

    def test_large_context_diff(self):
        """Test handling of diff with large context."""
        large_context = """--- a/large.py
+++ b/large.py
@@ -50,15 +50,15 @@ def large_function():
     # Context line 1
     # Context line 2
     # Context line 3
     # Context line 4
     # Context line 5
-    old_critical_line()
+    new_critical_line()
     # Context line 6
     # Context line 7
     # Context line 8
     # Context line 9
     # Context line 10"""

        result = answer_to_patch(large_context)
        validation = validate_diff_structure(result)

        assert validation["valid"], f"Large context diff invalid: {validation['error']}"
        assert validation["added_lines"] == 1
        assert validation["removed_lines"] == 1
        assert validation["context_lines"] == 10

        # Check that all context lines are properly categorized
        hunk = validation["hunks"][0]
        context_count = sum(
            1 for line_type, _ in hunk["lines"] if line_type == "context"
        )
        added_count = sum(1 for line_type, _ in hunk["lines"] if line_type == "added")
        removed_count = sum(
            1 for line_type, _ in hunk["lines"] if line_type == "removed"
        )

        assert context_count == 10
        assert added_count == 1
        assert removed_count == 1

    def test_index_header_with_permissions(self):
        """Test Index header with file permissions."""
        index_diff = """Index: src/main.py
===================================================================
--- a/src/main.py	(mode 100644)
+++ b/src/main.py	(mode 100755)
@@ -1,3 +1,4 @@
+#!/usr/bin/env python3
 def main():
     print("Hello")
     return 0"""

        result = answer_to_patch(index_diff)
        validation = validate_diff_structure(result)

        assert validation["valid"], (
            f"Index diff with permissions invalid: {validation['error']}"
        )
        assert validation["has_index_header"]
        assert validation["starts_properly"]
        assert "src/main.py" in validation["headers"][0][1]
        assert validation["added_lines"] == 1
        assert validation["removed_lines"] == 0

    def test_validator_comprehensive_edge_cases(self):
        """Test the structural validator with comprehensive edge cases."""

        # Test 1: Completely empty input
        empty_validation = validate_diff_structure("")
        assert not empty_validation["valid"]
        assert "Empty diff" in empty_validation["error"]

        # Test 2: Only headers, no hunks
        headers_only = """--- a/test.py
+++ b/test.py"""
        headers_validation = validate_diff_structure(headers_only)
        assert not headers_validation["valid"]
        assert "No valid hunks found" in headers_validation["error"]

        # Test 3: Missing +++ header
        missing_plus = """--- a/test.py
@@ -1,1 +1,1 @@
-old
+new"""
        missing_validation = validate_diff_structure(missing_plus)
        assert not missing_validation["valid"]
        assert "Missing +++ header" in missing_validation["error"]

        # Test 4: Multiple files in one diff
        multi_file = """--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-old1
+new1
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-old2
+new2"""
        multi_validation = validate_diff_structure(multi_file)
        # This should be valid - it's a proper multi-file diff
        assert multi_validation["valid"]
        # Note: Our validator currently treats this as one continuous diff
        # This is acceptable behavior for the current implementation
        assert multi_validation["added_lines"] >= 2
        assert multi_validation["removed_lines"] >= 2

        # Test 5: Diff with only additions
        only_adds = """--- /dev/null
+++ b/newfile.py
@@ -0,0 +1,3 @@
+def new_function():
+    return "hello"
+    # end"""
        adds_validation = validate_diff_structure(only_adds)
        assert adds_validation["valid"]
        assert adds_validation["added_lines"] == 3
        assert adds_validation["removed_lines"] == 0

        # Test 6: Diff with only deletions
        only_deletes = """--- a/oldfile.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_function():
-    return "goodbye"
-    # end"""
        deletes_validation = validate_diff_structure(only_deletes)
        assert deletes_validation["valid"]
        assert deletes_validation["added_lines"] == 0
        assert deletes_validation["removed_lines"] == 3


class TestHelperFunctions:
    """Test helper functions used by answer_to_patch."""

    def test_looks_like_unified_diff(self):
        """Test the _looks_like_unified_diff helper function."""
        from utils.benchmarks.swe_utils import _looks_like_unified_diff

        # Valid unified diff
        valid_diff = """--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old
+new"""
        assert _looks_like_unified_diff(valid_diff) is True

        # Git diff format
        git_diff = "diff --git a/file.py b/file.py"
        assert _looks_like_unified_diff(git_diff) is True

        # Invalid (no headers)
        invalid = "some random text"
        assert _looks_like_unified_diff(invalid) is False

        # Invalid (contains explanatory text)
        with_explanation = """--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old
+new
However, this change might cause issues."""
        assert _looks_like_unified_diff(with_explanation) is False
