import datetime
import os
import random
from typing import Optional, Set


class ChunkLockManager:
    """Manages chunk locks to prevent concurrent execution of the same chunk.
    Example usage:
    lock_manager = ChunkLockManager("/path/to/lock_dir")
    chunk_id = 0
    if lock_manager.acquire_lock(chunk_id):
        try:
            # Process the chunk
            pass
        finally:
            lock_manager.mark_chunk_completed(chunk_id)
    else:
        print(f"Chunk {chunk_id} is already being processed or completed.")
    """
    
    def __init__(self, lock_dir: str):
        self.lock_dir = lock_dir
        self.completion_dir = os.path.join(lock_dir, "completed")
        os.makedirs(self.lock_dir, exist_ok=True)
        os.makedirs(self.completion_dir, exist_ok=True)
        self.acquired_locks = set()

    def is_chunk_completed(self, chunk_id: int) -> bool:
        """Check if a chunk has been completed"""
        completion_file = os.path.join(self.completion_dir, f"chunk_{chunk_id}.completed")
        return os.path.exists(completion_file)
    

    def mark_chunk_completed(self, chunk_id: int, machine_id: Optional[str] = None) -> bool:
        """Mark a chunk as completed across all machines"""
        if machine_id is None:
            import socket
            machine_id = socket.gethostname()
            
        completion_file = os.path.join(self.completion_dir, f"chunk_{chunk_id}.completed")
        try:
            with open(completion_file, 'w') as f:
                f.write(f"{machine_id}\n{datetime.datetime.now().isoformat()}\n{os.getpid()}\n")
            return True
        except Exception as e:
            print(f"Error marking chunk {chunk_id} as completed: {e}")
            return False

    def get_completed_chunks(self) -> Set[int]:
        """Get set of all completed chunk IDs"""
        completed = set()
        try:
            for filename in os.listdir(self.completion_dir):
                if filename.startswith("chunk_") and filename.endswith(".completed"):
                    chunk_id = int(filename.replace("chunk_", "").replace(".completed", ""))
                    completed.add(chunk_id)
        except Exception as e:
            print(f"Error reading completed chunks: {e}")
        return completed

    def acquire_lock(self, chunk_id: int) -> bool:
        """Acquire a lock for the given chunk ID.
           If completed, return False.
           If already locked, return False.
           If successfully locked, return True.
        """

        if self.is_chunk_completed(chunk_id):
            return False

        lock_file = os.path.join(self.lock_dir, f"chunk_{chunk_id}.lock")
        try:
            # Try to create lock file with exclusive access
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

            # Write process info to lock file
            with os.fdopen(fd, 'w') as f:
                f.write(f"{os.getpid()}\n{datetime.datetime.now().isoformat()}\n")

            self.acquired_locks.add(chunk_id)
            return True

        except OSError:
            # Lock file already exists (chunk is taken)
            return False
        

    def get_next_available_chunk(self, total_chunks: int, exclude_chunks: Optional[Set[int]] = None) -> Optional[int]:
        """Get the next available chunk that's not locked or completed"""
        if exclude_chunks is None:
            exclude_chunks = set()

        # Get completed chunks from persistent storage
        completed_chunks = self.get_completed_chunks()

        # Create a list of available chunks and shuffle for randomness
        available_chunks = [i for i in range(total_chunks) 
                          if i not in exclude_chunks and i not in completed_chunks]
        random.shuffle(available_chunks)
        
        for chunk_id in available_chunks:
            if self.acquire_lock(chunk_id):
                return chunk_id
        
        return None

    def release_chunk(self, chunk_id: int):
        """Release a chunk lock"""
        if chunk_id in self.acquired_locks:
            lock_file = os.path.join(self.lock_dir, f"chunk_{chunk_id}.lock")
            if not os.path.exists(lock_file):
                self.acquired_locks.remove(chunk_id)
                return  # Lock file already removed
            try:
                os.remove(lock_file)
                self.acquired_locks.remove(chunk_id)
            except OSError:
                pass  # File might have been removed already

    
    def release_all_locks(self):
        """Release all acquired locks"""
        for chunk_id in list(self.acquired_locks):
            self.release_chunk(chunk_id)

    def is_chunk_locked(self, chunk_id: int) -> bool:
        """Check if a chunk is currently locked"""
        lock_file = os.path.join(self.lock_dir, f"chunk_{chunk_id}.lock")
        return os.path.exists(lock_file)


