#!/usr/bin/env python3
"""
Standalone test for filename sanitization security fix.
Can be run without dependencies.
"""
from pathlib import Path
import re


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent path traversal and other attacks.
    """
    if not filename:
        return "unnamed_file"
    
    # 1. Normalize path separators (handle Windows backslashes)
    filename = filename.replace('\\', '/')
    
    # 2. Get only the basename (strip any directory components)
    filename = Path(filename).name
    
    if not filename:
        return "unnamed_file"
    
    # 3. Remove or replace dangerous characters
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    
    # 4. Prevent dotfiles, but preserve extension if present
    if filename.startswith('.'):
        without_dots = filename.lstrip('.')
        if not '.' in without_dots and len(without_dots) <= 5 and without_dots:
            pass  # Keep leading dot (it's an extension)
        else:
            filename = without_dots
    
    # 5. Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            name = name[:max_length - len(ext) - 1]
            filename = f"{name}.{ext}"
        else:
            filename = filename[:max_length]
    
    # 6. Final check
    if not filename or filename == '.':
        return "unnamed_file"
    
    return filename


def run_tests():
    """Run security tests."""
    print("🔒 Testing Filename Sanitization Security Fix\n")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test cases: (input, expected_output, description)
    test_cases = [
        # Path traversal attacks
        ("../../etc/passwd", "passwd", "Path traversal (Unix)"),
        ("../../../api/main.py", "main.py", "Path traversal to app files"),
        ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe", "Path traversal (Windows)"),
        
        # Absolute paths
        ("/etc/passwd", "passwd", "Absolute path (Unix)"),
        ("C:\\Windows\\System32\\cmd.exe", "cmd.exe", "Absolute path (Windows)"),
        
        # Normal files (should work)
        ("document.pdf", "document.pdf", "Normal PDF file"),
        ("medical_form.pdf", "medical_form.pdf", "Normal underscore file"),
        ("report-2024.pdf", "report-2024.pdf", "Normal hyphen file"),
        
        # Spaces
        ("my document.pdf", "my_document.pdf", "Spaces replaced"),
        ("file with many spaces.txt", "file_with_many_spaces.txt", "Multiple spaces"),
        
        # Dangerous characters
        ("file;rm -rf.pdf", "filerm_-rf.pdf", "Command injection attempt (space→underscore)"),
        ("file|cmd.exe", "filecmd.exe", "Pipe character"),
        ("file<script>.pdf", "filescript.pdf", "HTML tags"),
        
        # Unicode characters
        ("file_🔥.pdf", "file_.pdf", "Unicode characters"),
        ("文档.pdf", ".pdf", "Unicode filename (extension preserved)"),
        
        # Dotfiles
        (".bashrc", "bashrc", "Leading dot removed"),
        ("..hidden", "hidden", "Double dots"),
        
        # Empty/invalid
        ("", "unnamed_file", "Empty string"),
        (".", "unnamed_file", "Single dot"),
        ("..", "unnamed_file", "Double dot"),
        
        # Extensions preserved
        ("archive.tar.gz", "archive.tar.gz", "Multiple extensions"),
        
        # Null bytes
        ("file\x00null.txt", "filenull.txt", "Null byte injection"),
    ]
    
    for input_val, expected, description in test_cases:
        result = _sanitize_filename(input_val)
        
        if result == expected:
            print(f"✅ PASS: {description}")
            print(f"   Input:    {repr(input_val)}")
            print(f"   Expected: {repr(expected)}")
            print(f"   Got:      {repr(result)}")
            tests_passed += 1
        else:
            print(f"❌ FAIL: {description}")
            print(f"   Input:    {repr(input_val)}")
            print(f"   Expected: {repr(expected)}")
            print(f"   Got:      {repr(result)}")
            tests_failed += 1
        print()
    
    print("=" * 60)
    print(f"\n📊 Results: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed == 0:
        print("\n🎉 All security tests PASSED! The fix is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {tests_failed} test(s) FAILED. Review the sanitization logic.")
        return 1


if __name__ == "__main__":
    exit(run_tests())

