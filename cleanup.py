"""
Cleanup utility for managing old analysis outputs.
Removes stale video files and JSON results based on configurable retention policies.
"""
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputCleaner:
    """Manages cleanup of processed video outputs and results."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        retention_days: int = 7,
        max_total_size_mb: int = 5000
    ):
        """
        Initialize the cleaner.
        
        Args:
            output_dir: Directory containing outputs (default: from config)
            retention_days: Keep files newer than this many days
            max_total_size_mb: Maximum total size of outputs directory in MB
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.retention_days = retention_days
        self.max_total_size_mb = max_total_size_mb
        self.max_total_size_bytes = max_total_size_mb * 1024 * 1024
    
    def get_file_age_days(self, filepath: Path) -> float:
        """Get the age of a file in days."""
        mtime = filepath.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / (24 * 60 * 60)
    
    def get_directory_size(self) -> int:
        """Get total size of output directory in bytes."""
        total = 0
        for f in self.output_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    
    def get_file_pairs(self) -> list:
        """
        Get list of file pairs (video + json) grouped by file_id.
        Returns list of dicts with file_id, paths, total size, and age.
        """
        pairs = {}
        
        for f in self.output_dir.iterdir():
            if not f.is_file():
                continue
            
            # Extract file_id from filename pattern: {file_id}_output.mp4 or {file_id}_results.json
            name = f.name
            if name.endswith("_output.mp4"):
                file_id = name.replace("_output.mp4", "")
                if file_id not in pairs:
                    pairs[file_id] = {"video": None, "json": None, "size": 0, "mtime": 0}
                pairs[file_id]["video"] = f
                pairs[file_id]["size"] += f.stat().st_size
                pairs[file_id]["mtime"] = max(pairs[file_id]["mtime"], f.stat().st_mtime)
            elif name.endswith("_results.json"):
                file_id = name.replace("_results.json", "")
                if file_id not in pairs:
                    pairs[file_id] = {"video": None, "json": None, "size": 0, "mtime": 0}
                pairs[file_id]["json"] = f
                pairs[file_id]["size"] += f.stat().st_size
                pairs[file_id]["mtime"] = max(pairs[file_id]["mtime"], f.stat().st_mtime)
        
        # Convert to list with file_id included
        result = []
        for file_id, data in pairs.items():
            data["file_id"] = file_id
            data["age_days"] = (time.time() - data["mtime"]) / (24 * 60 * 60)
            result.append(data)
        
        return result
    
    def cleanup_by_age(self, dry_run: bool = False) -> dict:
        """
        Remove files older than retention_days.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dict with deleted count and freed bytes
        """
        pairs = self.get_file_pairs()
        deleted_count = 0
        freed_bytes = 0
        
        for pair in pairs:
            if pair["age_days"] > self.retention_days:
                if dry_run:
                    logger.info(f"[DRY RUN] Would delete {pair['file_id']} (age: {pair['age_days']:.1f} days)")
                else:
                    if pair["video"] and pair["video"].exists():
                        pair["video"].unlink()
                        logger.info(f"Deleted: {pair['video'].name}")
                    if pair["json"] and pair["json"].exists():
                        pair["json"].unlink()
                        logger.info(f"Deleted: {pair['json'].name}")
                
                deleted_count += 1
                freed_bytes += pair["size"]
        
        return {"deleted_count": deleted_count, "freed_bytes": freed_bytes}
    
    def cleanup_by_size(self, dry_run: bool = False) -> dict:
        """
        Remove oldest files until directory is under max_total_size_mb.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dict with deleted count and freed bytes
        """
        current_size = self.get_directory_size()
        
        if current_size <= self.max_total_size_bytes:
            logger.info(f"Directory size ({current_size / 1024 / 1024:.1f} MB) is under limit ({self.max_total_size_mb} MB)")
            return {"deleted_count": 0, "freed_bytes": 0}
        
        # Sort by age (oldest first)
        pairs = sorted(self.get_file_pairs(), key=lambda x: x["mtime"])
        
        deleted_count = 0
        freed_bytes = 0
        
        for pair in pairs:
            if current_size <= self.max_total_size_bytes:
                break
            
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {pair['file_id']} to free {pair['size'] / 1024 / 1024:.1f} MB")
            else:
                if pair["video"] and pair["video"].exists():
                    pair["video"].unlink()
                if pair["json"] and pair["json"].exists():
                    pair["json"].unlink()
                logger.info(f"Deleted {pair['file_id']} to free space")
            
            current_size -= pair["size"]
            deleted_count += 1
            freed_bytes += pair["size"]
        
        return {"deleted_count": deleted_count, "freed_bytes": freed_bytes}
    
    def cleanup(self, dry_run: bool = False) -> dict:
        """
        Run all cleanup strategies.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dict with total deleted count and freed bytes
        """
        logger.info(f"Starting cleanup (dry_run={dry_run})")
        logger.info(f"  Retention: {self.retention_days} days")
        logger.info(f"  Max size: {self.max_total_size_mb} MB")
        
        # First, clean by age
        age_result = self.cleanup_by_age(dry_run)
        
        # Then, clean by size if still over limit
        size_result = self.cleanup_by_size(dry_run)
        
        total = {
            "deleted_count": age_result["deleted_count"] + size_result["deleted_count"],
            "freed_bytes": age_result["freed_bytes"] + size_result["freed_bytes"]
        }
        
        logger.info(f"Cleanup complete: {total['deleted_count']} items, {total['freed_bytes'] / 1024 / 1024:.1f} MB freed")
        return total


def main():
    """CLI entry point for cleanup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old analysis outputs")
    parser.add_argument("--retention-days", type=int, default=7, help="Keep files newer than N days (default: 7)")
    parser.add_argument("--max-size-mb", type=int, default=5000, help="Max directory size in MB (default: 5000)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    
    args = parser.parse_args()
    
    cleaner = OutputCleaner(
        retention_days=args.retention_days,
        max_total_size_mb=args.max_size_mb
    )
    
    cleaner.cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
