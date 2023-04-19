import subprocess as sp
from pathlib import Path
import logging
from typing import List

log = logging.getLogger(__name__)


class OneDrive:
    """Class to interact with OneDrive via the command line.

    Install onedrive by following https://linuxhint.com/install-microsoft-onedrive-ubuntu/ and then authenticating.

    """

    monitor_processes: List[sp.Popen] = []

    def __init__(self, syncdir: Path):
        self.syncdir = Path(syncdir).expanduser()
        if not self.syncdir.exists():
            raise ValueError(f"Syncdir {self.syncdir} does not exist.")

    def download(self, remote_dir: Path, resync: bool = False, skip: bool = False) -> Path:
        """Download a directory from OneDrive.

        Args:
            remote_dir (Path): Relative path to the directory to download.

        """
        args = [
            "onedrive",
            "--synchronize",
            "--download-only",
            "--syncdir",
            str(self.syncdir),
            "--single-directory",
            str(remote_dir),
            "--check-for-nosync",
        ]
        if resync:
            args.append("--resync")
        if not skip:
            log.info(f"Downloading {remote_dir}:\n{' '.join(args)}")
            sp.run(args, check=True)
            log.info("Download complete.")
        return self.syncdir / remote_dir

    def upload(self, remote_dir: Path, resync: bool = False):
        """Upload a directory to OneDrive.

        Args:
            remote_dir (Path): Relative path to the directory to upload.

        """
        args = [
            "onedrive",
            "--synchronize",
            "--upload-only",
            "--syncdir",
            str(self.syncdir),
            "--single-directory",
            str(remote_dir),
            "--no-remote-delete",
            "--check-for-nosync",
        ]
        if resync:
            args.append("--resync")
        log.debug(f"Uploading {remote_dir}:\n{' '.join(args)}")
        sp.run(args, check=True)
        log.debug("Upload complete.")

    def monitor(
        self,
        remote_dir: Path,
        monitor_interval: int = 10,
        no_remote_delete: bool = True,
        download_only: bool = True,
    ) -> Path:
        """Monitor a directory for changes.

        By default, only download new changes, but can be used to upload changes as well.

        Args:
            remote_dir (Path): Relative path to the directory to monitor.

        """
        args = [
            "onedrive",
            "--monitor",
            "--syncdir",
            str(self.syncdir),
            "--single-directory",
            str(remote_dir),
            "--monitor-interval",
            str(monitor_interval),
            "--check-for-nosync",
        ]
        if no_remote_delete:
            args.append("--no-remote-delete")
        if download_only:
            args.append("--download-only")
        log.info(f"Monitoring {remote_dir}:\n{' '.join(args)}")
        self.monitor_processes.append(sp.Popen(args))
        return self.syncdir / remote_dir

    def terminate(self):
        for process in self.monitor_processes:
            process.terminate()

    def __del__(self):
        self.terminate()
