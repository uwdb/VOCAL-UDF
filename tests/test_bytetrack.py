from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking, vis
import duckdb
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument(
    "--aspect_ratio_thresh", type=float, default=1000,
    help="threshold for filtering out boxes of which aspect ratio are above the given value."
)
parser.add_argument('--min_box_area', type=float, default=0, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

args = parser.parse_args()

conn = duckdb.connect()
conn.execute("CREATE TABLE Obj_clevrer (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float);")
conn.execute("COPY Obj_clevrer FROM '/home/enhao/EQUI-VOCAL/postgres/obj_clevrer.csv' (FORMAT 'csv', delimiter ',', header 0);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);")
df = conn.execute("SELECT fid, oid, x1, x2, y1, y2, 1 AS score FROM Obj_clevrer WHERE vid = 0").df()

img_height = 320
img_width = 480
aspect_ratio_thresh = 1.6
min_box_area = 10
cap = cv2.VideoCapture("/home/enhao/VOCAL-UDF/visprog/assets/clevrer1.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
fps = cap.get(cv2.CAP_PROP_FPS)

vid_writer = cv2.VideoWriter(
    "/home/enhao/VOCAL-UDF/tests/object_track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
)
vid_detection_writer = cv2.VideoWriter(
    "/home/enhao/VOCAL-UDF/tests/object_detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
)
tracker = BYTETracker(args)

# for image in images:
#    dets = detector(image)
#    online_targets = tracker.update(dets, info_imgs, img_size)

frame_id = 0
results = []
while True:
    ret_val, frame = cap.read()
    if ret_val:
        # outputs, img_info = predictor.inference(frame, timer)
        dets = df[df['fid'] == frame_id][['x1', 'y1', 'x2', 'y2', 'score']].values
        boxes = dets[:, :4]
        scores = dets[:, 4]
        cls_ids = np.zeros_like(scores)
        if dets is not None:
            online_targets = tracker.update(dets, [img_height, img_width], [img_height, img_width])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                # tlwh = t.tlwh
                tlwh = t._original_tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            online_im = plot_tracking(
                frame, online_tlwhs, online_ids, frame_id=frame_id + 1
            )
            online_detection_im = vis(frame, boxes, scores, cls_ids, class_names=['obj'])
        else:
            online_im = frame
            online_detection_im = frame
        vid_detection_writer.write(online_detection_im)
        vid_writer.write(online_im)
    else:
        break
    frame_id += 1
cap.release()
vid_writer.release()
vid_detection_writer.release()