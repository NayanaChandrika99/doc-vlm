"""Security tests for API endpoints."""
import pytest
from pathlib import Path
import sys

# Add api module to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import _sanitize_filename


class TestFilenameSanitization:
    """Test filename sanitization against path traversal attacks."""
    
    def test_path_traversal_basic(self):
        """Test basic path traversal attempts are blocked."""
        assert _sanitize_filename("../../etc/passwd") == "passwd"
        assert _sanitize_filename("../../../bad.txt") == "bad.txt"
        assert _sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    
    def test_absolute_paths(self):
        """Test absolute paths are converted to basename."""
        assert _sanitize_filename("/etc/passwd") == "passwd"
        assert _sanitize_filename("C:\\Windows\\System32\\cmd.exe") == "cmd.exe"
        assert _sanitize_filename("/usr/local/bin/malware") == "malware"
    
    def test_normal_filenames(self):
        """Test normal filenames pass through safely."""
        assert _sanitize_filename("document.pdf") == "document.pdf"
        assert _sanitize_filename("medical_form.pdf") == "medical_form.pdf"
        assert _sanitize_filename("report-2024.pdf") == "report-2024.pdf"
    
    def test_spaces_replaced(self):
        """Test spaces are replaced with underscores."""
        assert _sanitize_filename("my document.pdf") == "my_document.pdf"
        assert _sanitize_filename("file with spaces.txt") == "file_with_spaces.txt"
    
    def test_dangerous_characters_removed(self):
        """Test dangerous characters are stripped."""
        assert _sanitize_filename("file;rm -rf /.pdf") == "filerm-rf.pdf"
        assert _sanitize_filename("file|cmd.exe") == "filecmd.exe"
        assert _sanitize_filename("file<script>alert()</script>.pdf") == "filescriptalertscript.pdf"
        assert _sanitize_filename("file\x00null.txt") == "filenull.txt"
    
    def test_dotfile_prevention(self):
        """Test dotfiles are handled safely."""
        assert _sanitize_filename(".bashrc") == "bashrc"
        assert _sanitize_filename("..hidden") == "hidden"
        assert _sanitize_filename("...file") == "file"
    
    def test_empty_or_invalid(self):
        """Test empty or invalid filenames return default."""
        assert _sanitize_filename("") == "unnamed_file"
        assert _sanitize_filename(".") == "unnamed_file"
        assert _sanitize_filename("..") == "unnamed_file"
        assert _sanitize_filename("...") == "unnamed_file"
    
    def test_length_limiting(self):
        """Test very long filenames are truncated."""
        long_name = "a" * 300 + ".pdf"
        result = _sanitize_filename(long_name)
        assert len(result) <= 200
        assert result.endswith(".pdf")  # Extension preserved
    
    def test_extension_preservation(self):
        """Test file extensions are preserved when truncating."""
        long_name = "x" * 250 + ".pdf"
        result = _sanitize_filename(long_name)
        assert result.endswith(".pdf")
        assert len(result) <= 200
    
    def test_multiple_dots(self):
        """Test files with multiple dots are handled."""
        assert _sanitize_filename("archive.tar.gz") == "archive.tar.gz"
        assert _sanitize_filename("file.backup.2024.txt") == "file.backup.2024.txt"
    
    def test_unicode_characters(self):
        """Test unicode characters are removed, extension preserved."""
        assert _sanitize_filename("file_🔥.pdf") == "file_.pdf"
        assert _sanitize_filename("文档.pdf") == ".pdf"
    
    def test_null_bytes(self):
        """Test null bytes are removed (critical for some filesystems)."""
        assert _sanitize_filename("file\x00.txt") == "file.txt"
        assert _sanitize_filename("evil\x00.pdf.txt") == "evil.pdf.txt"
    
    def test_realistic_attack_vectors(self):
        """Test realistic attack scenarios."""
        # Path traversal to overwrite config
        assert _sanitize_filename("../../../config/settings.py") == "settings.py"
        
        # Path traversal to web root
        assert _sanitize_filename("../../../../var/www/html/shell.php") == "shell.php"
        
        # Mixed path separators
        assert _sanitize_filename("..\\../config\\..\\passwd") == "passwd"
        
        # Double encoding
        assert _sanitize_filename("..%2F..%2Fetc%2Fpasswd") == "..2F..2Fetc2Fpasswd"


class TestPathTraversalDefenseInDepth:
    """Test the secondary defense (path validation) after sanitization."""
    
    def test_sanitized_path_stays_in_upload_dir(self):
        """Test that even sanitized paths resolve within upload directory."""
        upload_dir = Path("datasets/uploads").resolve()
        job_id = "test-job-123"
        
        # Normal file
        safe_filename = _sanitize_filename("document.pdf")
        file_path = upload_dir / f"{job_id}_{safe_filename}"
        assert file_path.resolve().is_relative_to(upload_dir)
        
        # Attempted traversal (already sanitized)
        safe_filename = _sanitize_filename("../../../etc/passwd")
        file_path = upload_dir / f"{job_id}_{safe_filename}"
        assert file_path.resolve().is_relative_to(upload_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

