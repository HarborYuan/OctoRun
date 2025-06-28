import datetime
import os
import tempfile
import shutil
import socket
from unittest.mock import patch, mock_open
import pytest

from src.octorun.lock_manager import ChunkLockManager


class TestChunkLockManager:
    """Test suite for ChunkLockManager class"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.lock_manager = ChunkLockManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that __init__ creates necessary directories"""
        assert os.path.exists(self.lock_manager.lock_dir)
        assert os.path.exists(self.lock_manager.completion_dir)
        assert self.lock_manager.acquired_locks == set()
    
    def test_init_with_existing_directories(self):
        """Test initialization when directories already exist"""
        # Create another lock manager with the same directory
        lock_manager2 = ChunkLockManager(self.temp_dir)
        assert os.path.exists(lock_manager2.lock_dir)
        assert os.path.exists(lock_manager2.completion_dir)
    
    def test_is_chunk_completed_false_when_no_file(self):
        """Test is_chunk_completed returns False when completion file doesn't exist"""
        assert not self.lock_manager.is_chunk_completed(1)
    
    def test_is_chunk_completed_true_when_file_exists(self):
        """Test is_chunk_completed returns True when completion file exists"""
        chunk_id = 1
        completion_file = os.path.join(self.lock_manager.completion_dir, f"chunk_{chunk_id}.completed")
        
        # Create completion file
        with open(completion_file, 'w') as f:
            f.write("test")
        
        assert self.lock_manager.is_chunk_completed(chunk_id)
    
    def test_mark_chunk_completed_success(self):
        """Test successful chunk completion marking"""
        chunk_id = 1
        result = self.lock_manager.mark_chunk_completed(chunk_id, "test_machine")
        
        assert result is True
        assert self.lock_manager.is_chunk_completed(chunk_id)
        
        # Check file contents
        completion_file = os.path.join(self.lock_manager.completion_dir, f"chunk_{chunk_id}.completed")
        with open(completion_file, 'r') as f:
            lines = f.read().strip().split('\n')
            assert lines[0] == "test_machine"
            assert len(lines) == 3  # machine_id, timestamp, pid
    
    def test_mark_chunk_completed_default_machine_id(self):
        """Test mark_chunk_completed uses hostname when machine_id is None"""
        chunk_id = 1
        with patch('socket.gethostname', return_value='test_host'):
            result = self.lock_manager.mark_chunk_completed(chunk_id)
        
        assert result is True
        completion_file = os.path.join(self.lock_manager.completion_dir, f"chunk_{chunk_id}.completed")
        with open(completion_file, 'r') as f:
            lines = f.read().strip().split('\n')
            assert lines[0] == "test_host"
    
    def test_mark_chunk_completed_handles_exception(self):
        """Test mark_chunk_completed handles file write exceptions"""
        chunk_id = 1
        
        # Make completion directory read-only to cause write error
        os.chmod(self.lock_manager.completion_dir, 0o444)
        
        result = self.lock_manager.mark_chunk_completed(chunk_id, "test_machine")
        assert result is False
        
        # Restore permissions for cleanup
        os.chmod(self.lock_manager.completion_dir, 0o755)
    
    def test_get_completed_chunks_empty(self):
        """Test get_completed_chunks returns empty set when no chunks completed"""
        completed = self.lock_manager.get_completed_chunks()
        assert completed == set()
    
    def test_get_completed_chunks_with_completed_chunks(self):
        """Test get_completed_chunks returns correct chunk IDs"""
        # Mark some chunks as completed
        self.lock_manager.mark_chunk_completed(1, "machine1")
        self.lock_manager.mark_chunk_completed(3, "machine2")
        self.lock_manager.mark_chunk_completed(5, "machine3")
        
        completed = self.lock_manager.get_completed_chunks()
        assert completed == {1, 3, 5}
    
    def test_get_completed_chunks_handles_exception(self):
        """Test get_completed_chunks handles directory read exceptions"""
        # Remove completion directory to cause error
        shutil.rmtree(self.lock_manager.completion_dir)
        
        completed = self.lock_manager.get_completed_chunks()
        assert completed == set()
    
    def test_acquire_lock_success(self):
        """Test successful lock acquisition"""
        chunk_id = 1
        result = self.lock_manager.acquire_lock(chunk_id)
        
        assert result is True
        assert chunk_id in self.lock_manager.acquired_locks
        
        # Check lock file exists and contains correct info
        lock_file = os.path.join(self.lock_manager.lock_dir, f"chunk_{chunk_id}.lock")
        assert os.path.exists(lock_file)
        
        with open(lock_file, 'r') as f:
            lines = f.read().strip().split('\n')
            assert lines[0] == str(os.getpid())
            assert len(lines) == 2  # pid and timestamp
    
    def test_acquire_lock_fails_when_completed(self):
        """Test lock acquisition fails when chunk is already completed"""
        chunk_id = 1
        
        # Mark chunk as completed first
        self.lock_manager.mark_chunk_completed(chunk_id, "test_machine")
        
        result = self.lock_manager.acquire_lock(chunk_id)
        assert result is False
        assert chunk_id not in self.lock_manager.acquired_locks
    
    def test_acquire_lock_fails_when_already_locked(self):
        """Test lock acquisition fails when chunk is already locked"""
        chunk_id = 1
        
        # First acquisition should succeed
        result1 = self.lock_manager.acquire_lock(chunk_id)
        assert result1 is True
        
        # Create another lock manager instance
        lock_manager2 = ChunkLockManager(self.temp_dir)
        
        # Second acquisition should fail
        result2 = lock_manager2.acquire_lock(chunk_id)
        assert result2 is False
        assert chunk_id not in lock_manager2.acquired_locks
    
    def test_get_next_available_chunk_no_exclusions(self):
        """Test get_next_available_chunk without exclusions"""
        total_chunks = 5
        
        # Should get a chunk between 0 and 4
        chunk_id = self.lock_manager.get_next_available_chunk(total_chunks)
        assert chunk_id is not None
        assert 0 <= chunk_id < total_chunks
        assert chunk_id in self.lock_manager.acquired_locks
    
    def test_get_next_available_chunk_with_exclusions(self):
        """Test get_next_available_chunk with excluded chunks"""
        total_chunks = 5
        exclude_chunks = {0, 2, 4}
        
        chunk_id = self.lock_manager.get_next_available_chunk(total_chunks, exclude_chunks)
        assert chunk_id is not None
        assert chunk_id in {1, 3}
        assert chunk_id in self.lock_manager.acquired_locks
    
    def test_get_next_available_chunk_with_completed_chunks(self):
        """Test get_next_available_chunk skips completed chunks"""
        total_chunks = 5
        
        # Mark some chunks as completed
        self.lock_manager.mark_chunk_completed(0, "machine1")
        self.lock_manager.mark_chunk_completed(2, "machine2")
        
        chunk_id = self.lock_manager.get_next_available_chunk(total_chunks)
        assert chunk_id is not None
        assert chunk_id in {1, 3, 4}
        assert chunk_id in self.lock_manager.acquired_locks
    
    def test_get_next_available_chunk_all_taken(self):
        """Test get_next_available_chunk returns None when all chunks are taken"""
        total_chunks = 3
        
        # Mark all chunks as completed
        for i in range(total_chunks):
            self.lock_manager.mark_chunk_completed(i, f"machine{i}")
        
        chunk_id = self.lock_manager.get_next_available_chunk(total_chunks)
        assert chunk_id is None
    
    def test_get_next_available_chunk_randomness(self):
        """Test that get_next_available_chunk provides randomness"""
        total_chunks = 10
        results = set()
        
        # Run multiple times to check for randomness
        for _ in range(20):
            # Reset state
            self.lock_manager.release_all_locks()
            chunk_id = self.lock_manager.get_next_available_chunk(total_chunks)
            if chunk_id is not None:
                results.add(chunk_id)
        
        # Should get multiple different chunks due to randomness
        assert len(results) > 1
    
    def test_release_chunk_success(self):
        """Test successful chunk release"""
        chunk_id = 1
        
        # First acquire the lock
        self.lock_manager.acquire_lock(chunk_id)
        assert chunk_id in self.lock_manager.acquired_locks
        
        # Then release it
        self.lock_manager.release_chunk(chunk_id)
        assert chunk_id not in self.lock_manager.acquired_locks
        
        # Lock file should be removed
        lock_file = os.path.join(self.lock_manager.lock_dir, f"chunk_{chunk_id}.lock")
        assert not os.path.exists(lock_file)
    
    def test_release_chunk_not_acquired(self):
        """Test releasing chunk that wasn't acquired by this instance"""
        chunk_id = 1
        
        # Try to release without acquiring
        self.lock_manager.release_chunk(chunk_id)
        # Should not raise an exception
        assert chunk_id not in self.lock_manager.acquired_locks
    
    def test_release_chunk_handles_missing_file(self):
        """Test release_chunk handles case where lock file was already removed"""
        chunk_id = 1
        
        # Acquire lock
        self.lock_manager.acquire_lock(chunk_id)
        
        # Manually remove lock file
        lock_file = os.path.join(self.lock_manager.lock_dir, f"chunk_{chunk_id}.lock")
        os.remove(lock_file)
        
        # Release should handle missing file gracefully
        self.lock_manager.release_chunk(chunk_id)
        assert chunk_id not in self.lock_manager.acquired_locks
    
    def test_release_all_locks(self):
        """Test releasing all acquired locks"""
        chunk_ids = [1, 3, 5]
        
        # Acquire multiple locks
        for chunk_id in chunk_ids:
            self.lock_manager.acquire_lock(chunk_id)
        
        assert len(self.lock_manager.acquired_locks) == 3
        
        # Release all locks
        self.lock_manager.release_all_locks()
        assert len(self.lock_manager.acquired_locks) == 0
        
        # All lock files should be removed
        for chunk_id in chunk_ids:
            lock_file = os.path.join(self.lock_manager.lock_dir, f"chunk_{chunk_id}.lock")
            assert not os.path.exists(lock_file)
    
    def test_is_chunk_locked_true(self):
        """Test is_chunk_locked returns True when chunk is locked"""
        chunk_id = 1
        
        # Acquire lock
        self.lock_manager.acquire_lock(chunk_id)
        
        assert self.lock_manager.is_chunk_locked(chunk_id) is True
    
    def test_is_chunk_locked_false(self):
        """Test is_chunk_locked returns False when chunk is not locked"""
        chunk_id = 1
        
        assert self.lock_manager.is_chunk_locked(chunk_id) is False
    
    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access between two lock managers"""
        chunk_id = 1
        
        # Create two lock manager instances
        lock_manager1 = ChunkLockManager(self.temp_dir)
        lock_manager2 = ChunkLockManager(self.temp_dir)
        
        # First manager acquires lock
        result1 = lock_manager1.acquire_lock(chunk_id)
        assert result1 is True
        
        # Second manager tries to acquire same lock
        result2 = lock_manager2.acquire_lock(chunk_id)
        assert result2 is False
        
        # First manager releases lock
        lock_manager1.release_chunk(chunk_id)
        
        # Now second manager can acquire it
        result3 = lock_manager2.acquire_lock(chunk_id)
        assert result3 is True
        
        # Clean up
        lock_manager2.release_chunk(chunk_id)
    
    def test_workflow_complete_scenario(self):
        """Test complete workflow: acquire, work, complete, release"""
        total_chunks = 5
        
        # Get next available chunk
        chunk_id = self.lock_manager.get_next_available_chunk(total_chunks)
        assert chunk_id is not None
        
        # Simulate work being done...
        
        # Mark as completed
        result = self.lock_manager.mark_chunk_completed(chunk_id, "worker1")
        assert result is True
        
        # Release lock
        self.lock_manager.release_chunk(chunk_id)
        
        # Verify chunk is completed and lock is released
        assert self.lock_manager.is_chunk_completed(chunk_id)
        assert not self.lock_manager.is_chunk_locked(chunk_id)
        assert chunk_id not in self.lock_manager.acquired_locks
        
        # Trying to get same chunk again should skip it
        new_chunk = self.lock_manager.get_next_available_chunk(total_chunks)
        assert new_chunk != chunk_id  # Should get a different chunk
