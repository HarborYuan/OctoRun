import datetime
import os
import tempfile
import shutil
import subprocess
from unittest.mock import patch, mock_open, MagicMock, call

from src.octorun.runner import ProcessManager
from src.octorun.lock_manager import ChunkLockManager


class TestProcessManager:
    """Test suite for ProcessManager class"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.chunk_lock_dir = os.path.join(self.temp_dir, "chunk_locks")
        
        # Use the dummy script for testing
        dummy_script_path = os.path.join(os.path.dirname(__file__), 'dummy_script.py')
        
        self.config = {
            'script_path': dummy_script_path,
            'log_dir': self.log_dir,
            'chunk_lock_dir': self.chunk_lock_dir,
            'monitor_interval': 1,
            'restart_failed': True,
            'max_retries': 2
        }
    
    def teardown_method(self):
        """Clean up test environment after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.octorun.runner.ChunkLockManager')
    @patch('src.octorun.runner.datetime')
    def test_init(self, mock_datetime, mock_chunk_lock_manager):
        """Test ProcessManager initialization"""
        mock_start_time = datetime.datetime(2025, 6, 27, 10, 0, 0)
        mock_datetime.datetime.now.return_value = mock_start_time
        
        mock_lock_manager = MagicMock()
        mock_lock_manager.get_completed_chunks.return_value = {1, 2}
        mock_chunk_lock_manager.return_value = mock_lock_manager
        
        with patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            pm = ProcessManager(self.config)
            
            assert pm.config == self.config
            assert pm.processes == {}
            assert pm.chunk_lock_manager == mock_lock_manager
            assert pm.start_time == mock_start_time
            assert pm.completed_chunks == {1, 2}
            assert pm.failed_chunks == set()
            assert pm.retry_count == {}
            
            mock_log.assert_called_once_with("Found 2 previously completed chunks: [1, 2]")
    
    @patch('src.octorun.runner.os.makedirs')
    @patch('src.octorun.runner.datetime')
    @patch('builtins.open', new_callable=mock_open)
    def test_setup_logging(self, mock_file, mock_datetime, mock_makedirs):
        """Test setup_logging method"""
        mock_start_time = datetime.datetime(2025, 6, 27, 10, 30, 45)
        mock_datetime.datetime.now.return_value = mock_start_time
        mock_datetime.datetime.strftime = datetime.datetime.strftime
        
        with patch('src.octorun.runner.ChunkLockManager'), \
             patch.object(ProcessManager, 'log_message'):
            
            # Create ProcessManager - setup_logging will be called automatically
            pm = ProcessManager(self.config)
            
            # Verify the expected calls were made
            mock_makedirs.assert_called_once_with(self.log_dir, exist_ok=True)
            expected_session_log = os.path.join(self.log_dir, "session_20250627_103045.log")
            assert pm.session_log == expected_session_log
            
            # Check file was opened and written to
            mock_file.assert_called_once_with(expected_session_log, 'w')
            handle = mock_file.return_value.__enter__.return_value
            assert handle.write.call_count == 3

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.octorun.runner.datetime')
    def test_log_message(self, mock_datetime, mock_file):
        """Test log_message method"""
        mock_log_time = datetime.datetime(2025, 6, 27, 11, 0, 0)
        mock_datetime.datetime.now.return_value = mock_log_time
        mock_datetime.datetime.strftime = datetime.datetime.strftime
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch('builtins.print') as mock_print:
            
            # Make sure no completed chunks are returned to avoid log_message during init
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_completed_chunks.return_value = set()
            
            pm = ProcessManager(self.config)
            pm.session_log = "/tmp/session.log"
            
            # Reset the print mock to ignore any initialization calls
            mock_print.reset_mock()
            
            pm.log_message("Test message")
            
            expected_log = "[2025-06-27 11:00:00] Test message"
            mock_print.assert_called_once_with(expected_log)
            mock_file.assert_called_once_with("/tmp/session.log", 'a')
            handle = mock_file.return_value.__enter__.return_value
            handle.write.assert_called_once_with(expected_log + "\n")

    @patch('src.octorun.runner.subprocess.Popen')
    @patch('src.octorun.runner.os.environ')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.octorun.runner.datetime')
    def test_start_process_success(self, mock_datetime, mock_file, mock_environ, mock_popen):
        """Test successful process start"""
        mock_start_time = datetime.datetime(2025, 6, 27, 12, 0, 0)
        mock_datetime.datetime.now.return_value = mock_start_time
        
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_environ.copy.return_value = {'PATH': '/usr/bin'}
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            # Make sure no completed chunks are returned to avoid initial log_message
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_completed_chunks.return_value = set()
            
            pm = ProcessManager(self.config)
            result = pm.start_process(gpu_id=0, chunk_id=5, total_chunks=10)
            
            assert result == mock_process
            assert 5 in pm.processes
            
            proc_info = pm.processes[5]
            assert proc_info['process'] == mock_process
            assert proc_info['gpu_id'] == 0
            assert proc_info['start_time'] == mock_start_time
            assert proc_info['log_file'] == os.path.join(self.log_dir, "chunk_5.log")
            assert proc_info['status'] == 'running'
            
            dummy_script_path = os.path.join(os.path.dirname(__file__), 'dummy_script.py')
            expected_cmd = [
                'python', dummy_script_path,
                '--gpu_id', '0',
                '--chunk_id', '5',
                '--total_chunks', '10'
            ]
            mock_popen.assert_called_once_with(
                expected_cmd,
                stdout=mock_file.return_value,
                stderr=subprocess.STDOUT,
                env={'PATH': '/usr/bin'}
            )
            
            mock_log.assert_called_with("Started process for GPU 0, chunk 5 (PID: 12345)")
    
    @patch('src.octorun.runner.subprocess.Popen')
    @patch('builtins.open', new_callable=mock_open)
    def test_start_process_failure(self, mock_file, mock_popen):
        """Test process start failure"""
        mock_popen.side_effect = Exception("Failed to start process")
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            # Make sure no completed chunks are returned to avoid initial log_message
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_completed_chunks.return_value = set()
            
            pm = ProcessManager(self.config)
            result = pm.start_process(gpu_id=0, chunk_id=5, total_chunks=10)
            
            assert result is None
            assert 5 not in pm.processes
            # Check that log_message was called with an error message containing our exception
            mock_log.assert_any_call("Error starting process for GPU 0, chunk 5: Failed to start process")
    
    @patch('src.octorun.runner.datetime')
    def test_check_processes_running(self, mock_datetime):
        """Test check_processes with running processes"""
        mock_current_time = datetime.datetime(2025, 6, 27, 13, 0, 0)
        mock_datetime.datetime.now.return_value = mock_current_time
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message'):
            
            pm = ProcessManager(self.config)
            
            # Mock running process
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            
            pm.processes[1] = {
                'process': mock_process,
                'gpu_id': 0,
                'start_time': mock_current_time,
                'log_file': '/tmp/chunk_1.log',
                'status': 'running'
            }
            
            status = pm.check_processes()
            
            assert status['running'] == [1]
            assert status['completed'] == []
            assert status['failed'] == []
    
    @patch('src.octorun.runner.datetime')
    def test_check_processes_completed(self, mock_datetime):
        """Test check_processes with completed processes"""
        mock_start_time = datetime.datetime(2025, 6, 27, 12, 0, 0)
        mock_end_time = datetime.datetime(2025, 6, 27, 13, 0, 0)
        mock_datetime.datetime.now.return_value = mock_end_time
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_completed_chunks.return_value = set()
            
            pm = ProcessManager(self.config)
            
            # Mock completed process
            mock_process = MagicMock()
            mock_process.poll.return_value = 0  # Completed successfully
            mock_process.returncode = 0
            
            pm.processes[1] = {
                'process': mock_process,
                'gpu_id': 0,
                'start_time': mock_start_time,
                'log_file': '/tmp/chunk_1.log',
                'status': 'running'
            }
            
            status = pm.check_processes()
            
            assert status['running'] == []
            assert status['completed'] == [1]
            assert status['failed'] == []
            assert 1 in pm.completed_chunks
            assert pm.processes[1]['status'] == 'completed'
            
            mock_lock_manager.mark_chunk_completed.assert_called_once_with(1)
            mock_lock_manager.release_chunk.assert_called_once_with(1)
            mock_log.assert_called_once_with("GPU 0, chunk 1 completed successfully (Duration: 1:00:00)")
    
    @patch('src.octorun.runner.datetime')
    def test_check_processes_failed_with_retry(self, mock_datetime):
        """Test check_processes with failed processes that should be retried"""
        mock_current_time = datetime.datetime(2025, 6, 27, 13, 0, 0)
        mock_datetime.datetime.now.return_value = mock_current_time
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            pm = ProcessManager(self.config)
            
            # Mock failed process
            mock_process = MagicMock()
            mock_process.poll.return_value = 1  # Failed
            mock_process.returncode = 1
            
            pm.processes[1] = {
                'process': mock_process,
                'gpu_id': 0,
                'start_time': mock_current_time,
                'log_file': '/tmp/chunk_1.log',
                'status': 'running'
            }
            
            status = pm.check_processes()
            
            assert status['running'] == []
            assert status['completed'] == []
            assert status['failed'] == [1]
            assert 1 in pm.failed_chunks
            assert pm.processes[1]['status'] == 'failed'
            assert pm.retry_count[1] == 1
            
            mock_lock_manager.release_chunk.assert_called_once_with(1)
            expected_calls = [
                call("GPU 0, chunk 1 failed with return code 1"),
                call("Scheduling retry 1/2 for chunk 1")
            ]
            mock_log.assert_has_calls(expected_calls)
    
    def test_get_completion_status(self):
        """Test get_completion_status method"""
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message'):
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_completed_chunks.return_value = {1, 2, 3}
            
            pm = ProcessManager(self.config)
            pm.processes = {
                4: {'status': 'running'},
                5: {'status': 'running'},
                6: {'status': 'completed'}
            }
            
            status = pm.get_completion_status(total_chunks=10)
            
            assert status['completed_count'] == 3
            assert status['running_count'] == 2
            assert status['remaining_count'] == 5
            assert status['completion_percentage'] == 30.0
            assert status['completed_chunks'] == [1, 2, 3]
            assert status['running_chunks'] == [4, 5]
    
    @patch('src.octorun.runner.time.sleep')
    @patch.object(ProcessManager, 'start_process')
    @patch.object(ProcessManager, 'check_processes')
    @patch.object(ProcessManager, 'get_completion_status')
    @patch.object(ProcessManager, 'print_summary')
    def test_run_complete_workflow(self, mock_print_summary, mock_get_completion_status, 
                                  mock_check_processes, mock_start_process, mock_sleep):
        """Test the main run method with a complete workflow"""
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            
            # Setup mock returns for chunk assignment
            mock_lock_manager.get_next_available_chunk.side_effect = [0, 1, None, None]
            
            # Setup completion status - first incomplete, then complete
            mock_get_completion_status.side_effect = [
                {'completed_count': 1, 'completion_percentage': 50.0, 'running_chunks': [1]},
                {'completed_count': 2, 'completion_percentage': 100.0, 'running_chunks': []}
            ]
            
            # Setup process status
            mock_check_processes.return_value = {
                'running': [1],
                'completed': [0],
                'failed': []
            }
            
            pm = ProcessManager(self.config)
            pm.run(gpu_ids=[0, 1], total_chunks=2)
            
            # Verify initial process starts
            assert mock_start_process.call_count == 2
            mock_start_process.assert_any_call(0, 0, 2, kwargs=None)
            mock_start_process.assert_any_call(1, 1, 2, kwargs=None)
            
            # Verify monitoring loop
            mock_check_processes.assert_called_once()
            mock_print_summary.assert_called_once()
    
    def test_cleanup(self):
        """Test cleanup method"""
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            pm = ProcessManager(self.config)
            
            # Mock running process
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            mock_process.pid = 12345
            
            pm.processes[1] = {
                'process': mock_process,
                'gpu_id': 0,
                'start_time': datetime.datetime.now(),
                'log_file': '/tmp/chunk_1.log',
                'status': 'running'
            }
            
            pm.cleanup()
            
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once_with(timeout=10)
            mock_lock_manager.release_all_locks.assert_called_once()
            
            expected_calls = [
                call("Terminating process for chunk 1 (PID: 12345)"),
                call("All chunk locks released")
            ]
            mock_log.assert_has_calls(expected_calls)
    
    @patch('src.octorun.runner.datetime')
    def test_print_summary(self, mock_datetime):
        """Test print_summary method"""
        mock_start_time = datetime.datetime(2025, 6, 27, 10, 0, 0)
        mock_end_time = datetime.datetime(2025, 6, 27, 12, 0, 0)
        mock_datetime.datetime.now.return_value = mock_end_time
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_completed_chunks.return_value = {1, 2, 3, 4}
            
            pm = ProcessManager(self.config)
            pm.start_time = mock_start_time
            pm.completed_chunks = {2, 3}
            pm.failed_chunks = {5, 6}
            pm.session_log = "/tmp/session.log"
            
            pm.print_summary()
            
            expected_calls = [
                call("\n" + "=" * 80),
                call("FINAL SUMMARY"),
                call("=" * 80),
                call("Total execution time: 2:00:00"),
                call("Completed chunks (globally): 4"),
                call("Completed chunks (this session): 2"),
                call("Failed chunks (this session): 2"),
                call("Failed chunk IDs: [5, 6]"),
                call("Session log: /tmp/session.log"),
                call(f"Individual logs: {self.log_dir}/chunk_*.log")
            ]
            mock_log.assert_has_calls(expected_calls)
    
    @patch.object(ProcessManager, 'cleanup')
    @patch.object(ProcessManager, 'print_summary')
    @patch('src.octorun.runner.time.sleep')
    def test_run_keyboard_interrupt(self, mock_sleep, mock_print_summary, mock_cleanup):
        """Test run method handles KeyboardInterrupt"""
        mock_sleep.side_effect = KeyboardInterrupt("User interrupted")
        
        with patch('src.octorun.runner.ChunkLockManager') as mock_chunk_lock_manager, \
             patch.object(ProcessManager, 'setup_logging'), \
             patch.object(ProcessManager, 'log_message') as mock_log:
            
            mock_lock_manager = mock_chunk_lock_manager.return_value
            mock_lock_manager.get_next_available_chunk.return_value = None
            
            pm = ProcessManager(self.config)
            pm.run(gpu_ids=[0], total_chunks=1)
            
            mock_log.assert_any_call("Interrupted by user. Terminating processes...")
            mock_cleanup.assert_called_once()
            mock_print_summary.assert_called_once()


class TestProcessManagerIntegration:
    """Integration tests for ProcessManager with real ChunkLockManager"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.chunk_lock_dir = os.path.join(self.temp_dir, "chunk_locks")
        
        # Create a simple test script that exits successfully
        self.test_script = os.path.join(os.path.dirname(__file__), 'dummy_script.py')
        
        self.config = {
            'script_path': self.test_script,
            'log_dir': self.log_dir,
            'chunk_lock_dir': self.chunk_lock_dir,
            'monitor_interval': 0.1,  # Short interval for testing
            'restart_failed': False,
            'max_retries': 1
        }
    
    def teardown_method(self):
        """Clean up test environment after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_integration_with_real_chunk_lock_manager(self):
        """Test ProcessManager with real ChunkLockManager"""
        with patch.object(ProcessManager, 'log_message'):
            pm = ProcessManager(self.config)
            
            # Verify ChunkLockManager is properly initialized
            assert isinstance(pm.chunk_lock_manager, ChunkLockManager)
            assert os.path.exists(pm.chunk_lock_manager.lock_dir)
            assert os.path.exists(pm.chunk_lock_manager.completion_dir)
    
    def test_integration_process_lifecycle(self):
        """Test complete process lifecycle with real subprocess"""
        with patch.object(ProcessManager, 'log_message'):
            pm = ProcessManager(self.config)
            
            # Start a simple echo process
            process = pm.start_process(gpu_id=0, chunk_id=1, total_chunks=2)
            
            assert process is not None
            assert 1 in pm.processes
            
            # Wait for process to complete
            process.wait()
            
            # Check process status
            status = pm.check_processes()
            
            # Echo should complete successfully
            assert status['completed'] == [1]
            assert 1 in pm.completed_chunks
            
            # Verify chunk was marked as completed
            assert pm.chunk_lock_manager.is_chunk_completed(1)
