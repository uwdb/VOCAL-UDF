import cv2
import logging
import pandas as pd
from vocaludf.utils import (
    expand_box,
)
from pathlib import Path
from typing import Optional
from rich.prompt import Prompt
from rich.panel import Panel
from rich import print as rprint   # avoid clobbering built-in print

from vocaludf.udf_selector import UDFSelector

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Interactive selector
# --------------------------------------------------------------------------- #
class InteractiveUDFSelector(UDFSelector):
    def _ask_binary_label(self, img_path: Path, n_obj: int) -> bool:
        """
        Ask the user to label *img_path*.
            1 → positive / true
            0 → negative / false

        Because a plain terminal can't render the image, we just tell the user
        where it lives (open it with your favourite viewer) and wait for input.
        """
        obj_vars = [v.strip() for v in self.udf_signature.split('(')[1].split(')')[0].split(',')]
        if n_obj == 1:
            instruction = f"The object {obj_vars[0]} is highlighted in red box."
        else:
            instruction = f"The object {obj_vars[0]} is highlighted in red box, and the object {obj_vars[1]} is highlighted in blue box."

        rprint(
            Panel(f"""
Please label the following image. {instruction}
UDF signature: {self.udf_signature}
UDF description: {self.udf_description}
Open the image below in any viewer, then come back here and type [bold]1[/bold] (positive) or [bold]0[/bold] (negative):
[underline]{img_path}[/underline]
        """))

        while True:
            ans = Prompt.ask("Label (1 = positive, 0 = negative)").strip()
            if ans == "1":
                return True
            if ans == "0":
                return False
            rprint("[red]Please type 1 or 0.[/red]")


    def request_label(self, df: pd.DataFrame, n_obj: int, gt_udf_name: Optional[str]) -> int:  # noqa: D401
        """
        Blockingly ask the human for a binary label for *row*.

        Returns
        -------
        int
            1 → positive / true
            0 → negative / false
        """
        assert len(df) == 1, "InteractiveUDFSelector.request_label() expects a single-row DataFrame"
        row = df.iloc[0]
        vid = row['vid']
        fid = row['fid']
        frame = self.frame_processing_for_program(vid, fid)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_size = frame.shape[:2]
        # Draw bounding box on the frame and save it to a temporary file
        if n_obj == 1:
            x1, y1, x2, y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            cv2.rectangle(frame, (o1_x1, o1_y1), (o1_x2, o1_y2), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (o2_x1, o2_y1), (o2_x2, o2_y2), color=(255, 0, 0), thickness=1)
        img_path = Path(self.shared_resources.interactive_labeling_dir) / f"frame_{vid}_{fid}.jpg"
        cv2.imwrite(str(img_path), frame)

        label: bool = self._ask_binary_label(img_path, n_obj)
        return label
