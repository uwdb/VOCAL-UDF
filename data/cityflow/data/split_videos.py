# /gscratch/balazinska/enhaoz/AIC23_NLRetrieval_HCMIU_CVIP/data/json/dataclean_v1/train_standard.json
# {
#     "b06c903c-a25d-45fe-b0d5-294f72e34023": {
#         "frames": [
#             "./validation/S02/c006/img1/000001.jpg",
#             "./validation/S02/c006/img1/000002.jpg",
#             "./validation/S02/c006/img1/000003.jpg",
#             "./validation/S02/c006/img1/000004.jpg",
#             "./validation/S02/c006/img1/000005.jpg",
#             "./validation/S02/c006/img1/000006.jpg",
#             "./validation/S02/c006/img1/000007.jpg"
#         ],
#         "boxes": [
#             [
#                 539,
#                 606,
#                 273,
#                 277
#             ],
#             [
#                 532,
#                 631,
#                 271,
#                 282
#             ],
#             [
#                 526,
#                 657,
#                 270,
#                 287
#             ],
#             [
#                 516,
#                 665,
#                 283,
#                 309
#             ],
#             [
#                 512,
#                 703,
#                 284,
#                 317
#             ],
#             [
#                 496,
#                 713,
#                 303,
#                 341
#             ],
#             [
#                 481,
#                 752,
#                 317,
#                 328
#             ]
#         ],
#         "nl": [
#             "red sedan go straight",
#             "red medium sedan go straight",
#             "red vehicle go straight"
#         ],
#     },
# }

import json
import os
import pandas as pd

# Load the JSON file
json_file = '/gscratch/balazinska/enhaoz/AIC23_NLRetrieval_HCMIU_CVIP/data/json/dataclean_v1/train_standard.json'
with open(json_file, 'r') as f:
    data = json.load(f)

old_vnames = {}
old_fnames = set()
for k, v in data.items():
    for frame in v['frames']:
        old_vname = '/'.join(frame.split('/')[1:4])
        old_fid = int(frame.split('/')[-1].split('.')[0])
        if old_vname not in old_vnames:
            old_vnames[old_vname] = old_fid
        else:
            old_vnames[old_vname] = max(old_vnames[old_vname], old_fid)
        old_fnames.add(frame)

print(len(old_vnames), len(old_fnames))

vname_to_offset = {}
offset = 0
for vname in sorted(old_vnames):
    vname_to_offset[vname] = offset
    offset += (old_vnames[vname] - 1) // 50 + 1

frame_to_vid_fid = {}
data = []
for frame in old_fnames:
    vname = '/'.join(frame.split('/')[1:4])
    fid = int(frame.split('/')[-1].split('.')[0])
    offset = vname_to_offset[vname]
    frame_to_vid_fid[frame] = {"vid": offset + (fid - 1) // 50, "fid": (fid - 1) % 50}
    data.append((frame, frame_to_vid_fid[frame]["vid"], frame_to_vid_fid[frame]["fid"]))
new_vids = set()
for v in frame_to_vid_fid.values():
    new_vids.add(v["vid"])

print(len(new_vids))

# Save frame_to_vid_fid to a JSON file
new_json_file = '/gscratch/balazinska/enhaoz/VOCAL-UDF/data/cityflow/data/fname_to_vid_fid.json'
with open(new_json_file, 'w') as f:
    json.dump(frame_to_vid_fid, f, indent=4)

metadata_df = pd.DataFrame(data, columns=["fname", "vid", "fid"])
# sort by vid and fid
metadata_df = metadata_df.sort_values(by=["vid", "fid"]).reset_index(drop=True)
metadata_df.to_csv('/gscratch/balazinska/enhaoz/VOCAL-UDF/data/cityflow/data/fname_to_vid_fid.csv', index=False)