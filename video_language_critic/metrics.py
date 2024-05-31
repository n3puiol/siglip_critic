from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import collections
import json
import operator
import os
import pickle
import numpy as np
import torch
import cv2

from sklearn.metrics import roc_auc_score

VIDEO_PATH = f"{os.environ['WORK']}/nlr/vlm_dataset/pick_videos"
LOG_PATH = "./logs"

NEGATIVE_TASKS = (
    "Perform failed grasp",
    "No caption",
    "Do nothing",
)


def play_video(
    video_name,
    display=False,
    save_name=None,
    caption=None,
    log_path=None,
    start_time=None,
    end_time=None,
    similarity=-1,
):
    if start_time is not None or end_time is not None:
        assert (
            isinstance(start_time, int)
            and isinstance(end_time, int)
            and start_time > -1
            and end_time > start_time
        ), "Must pass a valid start_time and end_time"

    video_path = f"{VIDEO_PATH}/{video_name}.avi"
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration

    if start_time is not None:
        start_sec, end_sec = (
            start_time,
            end_time if end_time <= total_duration else total_duration,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    if log_path:
        assert save_name is not None, "Must provide a save_name"
        # Create a VideoWriter object to save the new video
        out = cv2.VideoWriter(
            f"{log_path}/{save_name}.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height),
        )
    # Go through the frames of the video
    for sec in np.arange(start_sec, end_sec + 1):
        sec_base = int(sec * fps)

        for sub_index in np.arange(fps):
            frame_idx = sec_base + sub_index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            # print(f"FPS: {frame}")
            # Read a frame from the video
            ret, frame = cap.read()

            # Check if the frame is read successfully
            if not ret:
                break

            if caption:
                # Add the caption to the frame
                if start_time is not None:
                    text = f"{caption}. Time: {frame_idx/fps:.2f}s, frame: {frame_idx}. Similarity: {similarity:.2f}"
                else:
                    text = caption
                cv2.putText(
                    frame,
                    text,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
            if log_path:
                if start_time is not None:
                    # slow motion by appending same frame n times:
                    repeat = 1
                else:
                    repeat = 1
                for _ in range(repeat):
                    out.write(frame)

            if display:
                # Display the frame
                cv2.imshow("Video", frame)

            # Wait for 25 milliseconds and check if the user wants to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    # Release the video file and close the window
    cap.release()
    if log_path:
        out.release()

    return frame


def compute_metrics(
    x,
    video_ids=None,
    captions=None,
    labels=None,
):
    sx = np.sort(-x, axis=1)

    if labels is None:
        correct_pair_scores = np.diag(-x)
    else:
        correct_pair_scores = -x[labels.astype(bool)]
        sx = sx[np.any(labels, axis=1)]
    correct_pair_scores = correct_pair_scores[:, np.newaxis]
    # Need to drop rows of sx where there is no label.
    ind = sx - correct_pair_scores
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}

    SIMILARITY_THRESHOLD = (
        # -25
        -30  # Taken by looking at the R@5 result. With this threhold, the answer seems to be mostly correct
    )

    metrics["R1"] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics["R5"] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics["R10"] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics["MedianR"] = np.median(ind) + 1
    metrics["MeanR"] = np.mean(ind) + 1
    # metrics["cols"] = [int(i) for i in list(ind)]

    if (video_ids is not None) and (captions is not None):
        # During testing, plot
        all_captions = []
        for caption in captions:
            all_captions.extend([sub_cap for sub_cap in caption])

        all_videos = []
        start_times = []
        end_times = []
        for video_id in video_ids:
            if isinstance(video_id, str):
                all_videos.extend([sub_cap for sub_cap in video_id])
            elif isinstance(video_id, dict):
                all_videos.extend([sub_cap for sub_cap in video_id["video_id"]])
                start_times.extend([start for start in video_id["start_time"]])
                end_times.extend([end for end in video_id["end_time"]])
            elif isinstance(video_id, tuple):
                all_videos.extend([sub_cap for sub_cap in video_id])
            else:
                raise ValueError("Video ID must be a string or a dictionary")
        sx_idx = np.argsort(-x, axis=1)
        for idx, caption in enumerate(all_captions):
            # print(f"Videos with high simlarity (< {SIMILARITY_THRESHOLD}) for caption")
            # print(caption)
            for sub_idx in range(sx_idx.shape[0]):
                similarity = sx[idx, sub_idx]
                if similarity > SIMILARITY_THRESHOLD or sub_idx > 4:
                    continue
                extra = "_true" if sub_idx == ind[idx] else ""

                video_idx = sx_idx[idx, sub_idx]

                start_time = None
                end_time = None
                if start_times and end_times:
                    start_time = int(start_times[video_idx])
                    end_time = int(end_times[video_idx])
                    extra += f"_time{start_time}-{end_time}"

                save_name = f"test_video{idx}_sim{sub_idx}{extra}"
                play_video(
                    all_videos[video_idx],
                    save_name=save_name,
                    caption=caption,
                    log_path=f"{LOG_PATH}",
                    start_time=start_time,
                    end_time=end_time,
                    display=False,
                    similarity=similarity,
                )
                # Display the frame
                cv2.destroyAllWindows()
        with open(f"{LOG_PATH}/metrics.json", "w") as f:
            json.dump(metrics, f)

    return metrics


def print_computed_metrics(metrics):
    r1 = metrics["R1"]
    r5 = metrics["R5"]
    r10 = metrics["R10"]
    mr = metrics["MR"]
    print(
        "R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}".format(
            r1, r5, r10, mr
        )
    )


# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k=[1, 5, 10]):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
    mask = ~torch.logical_or(
        torch.isinf(permuted_original_data), torch.isnan(permuted_original_data)
    )
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    # assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
        valid_ranks = torch.tensor(valid_ranks)
    results = {
        f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks))
        for k in top_k
    }
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results["MR"] = results["MedianR"]
    return results


def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float("-inf")
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T


def compute_symmetric_loss(
    args, model, sim_tensor, labels=None, captions=None, device=None, video_mask=None
):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    if not torch.is_tensor(video_mask):
        video_mask = torch.tensor(video_mask)
    if device is not None:
        sim_tensor = sim_tensor.to(device)
        labels = labels.to(device)
        video_mask = video_mask.to(device)
    loss = 0
    breakdown = {}
    if args.loss_type == "sequence_ranking_loss":
        ranking_loss, breakdown1 = model.ranking_loss_fct(
            sim_tensor, labels, video_mask
        )
        ranking_loss *= args.ranking_loss_weight
        loss += ranking_loss
        breakdown.update({f"tv_{k}": v for k, v in breakdown1.items()})
        sim_tensor = sim_tensor[:, :, -1]
    if args.loss_type == "binary_cross_entropy":
        loss, breakdown1 = model.loss_fct(sim_tensor, labels, captions)
        breakdown.update({f"tv_{k}": v for k, v in breakdown1.items()})
    else:
        labels_T = labels.T if labels is not None else None
        loss1, breakdown1 = model.loss_fct(sim_tensor, labels)
        loss2, breakdown2 = model.loss_fct(sim_tensor.T, labels_T)
        breakdown.update({f"tv_{k}": v for k, v in breakdown1.items()})
        breakdown["tv_loss"] = loss1
        breakdown["vt_loss"] = loss2
        ce_loss = (loss1 + loss2) / 2
        if loss != 0:
            breakdown["ce_loss"] = ce_loss
        loss += ce_loss
        breakdown.update({f"vt_{k}": v for k, v in breakdown2.items()})
    return loss.cpu().detach().numpy(), breakdown


def get_auc_per_task(sim_matrix, captions, labels):
    """Get ROC-AUC averaged per task, with all other tasks as negative examples."""
    task_to_auc = {}
    video_has_label = np.any(labels, axis=0)
    # Only average over caption types that match videos (leaves out negative captions).
    # Corresponds to the first dimension of sim_matrix.
    included_captions = [
        caption for v, caption in enumerate(captions) if video_has_label[v]
    ]
    tasks = set(included_captions)
    for task in tasks:
        # Idxs of all captions == 'task'.
        idxs = [i for i, c in enumerate(included_captions) if c == task]
        # sim_matrix[idx] is an identical row for each idx in idxs: use the first.
        task_scores = sim_matrix[idxs[0]]
        task_labels = [c == task for c in captions]
        task_to_auc[task] = roc_auc_score(task_labels, task_scores)
    return np.mean(list(task_to_auc.values()))


LABELED_NEGATIVE_TASKS = {
    "vlm": [
        [  # color
            "Grasp the azure object",
            "Grasp the black object",
            "Grasp the blue object",
            "Grasp the brown object",
            "Grasp the chocolate object",
            "Grasp the coral object",
            "Grasp the cyan object",
            "Grasp the gold object",
            "Grasp the gray object",
            "Grasp the green object",
            "Grasp the lime object",
            "Grasp the magenta object",
            "Grasp the maroon object",
            "Grasp the navy object",
            "Grasp the olive object",
            "Grasp the orange object",
            "Grasp the pink object",
            "Grasp the purple object",
            "Grasp the red object",
            "Grasp the rose object",
            "Grasp the silver object",
            "Grasp the teal object",
            "Grasp the violet object",
            "Grasp the white object",
            "Grasp the yellow object",
        ],
        [  # shape
            "Grasp the cross-shaped block",
            "Grasp the cube",
            "Grasp the cylinder",
            "Grasp the flower",
            "Grasp the letter of 't'",
            "Grasp the moon",
            "Grasp the star",
            "Grasp the triangular prism",
        ],
        [  # relative position (front-back)
            "Grasp the front object",
            "Grasp the rear object",
        ],
        [  # relative position (left-right)
            "Grasp the left object",
            "Grasp the right object",
        ],
        [  # size
            "Grasp the larger object",
            "Grasp the smaller object",
        ],
    ],
    # Only included tasks with truly the same starting state (e.g. open and close drawer
    # do not have the same initial state, although the objects in the scene are the same).
    "mw": [
        [
            "Push a mug under a coffee machine",  # "coffee-push-v2",
            "Push a button on the coffee machine",  # "coffee-button-v2",
        ],
        [
            "Rotate the faucet clockwise",  # "faucet-open-v2",
            "Rotate the faucet counter-clockwise",  # "faucet-close-v2",
        ],
        [
            "Pull a puck to a goal",  # "push-back-v2",
            "Push the puck to a goal",  # "push-v2",
        ],
        [
            "Grasp a stick and pull a box with the stick",  # "stick-pull-v2",
            "Grasp a stick and push a box using the stick",  # "stick-push-v2",
        ],
        [
            "Lock the door by rotating the lock clockwise",  # "door-lock-v2",
            "Open a door with a revolving joint",  # "door-open-v2",
        ],
    ],
}


def get_labeled_auc_per_task(
    sim_matrix,
    captions,
    labels,
    datatype="vlm",
    global_negative_tasks=None,
):
    def _get_negatives_for_task(task):
        neg_tasks = []
        groups = [ts for ts in LABELED_NEGATIVE_TASKS[datatype] if task in ts]
        for group in groups:
            group_items = group.copy()
            group_items.remove(task)
            neg_tasks.extend(group_items)
        return list(set(neg_tasks))

    if global_negative_tasks is None:
        global_negative_tasks = NEGATIVE_TASKS
    task_to_auc = {}
    video_has_label = np.any(labels, axis=0)
    # Only average over caption types that match videos (leaves out negative captions).
    # Corresponds to the first dimension of sim_matrix.
    captions_to_consider = [
        caption for v, caption in enumerate(captions) if video_has_label[v]
    ]
    tasks = set(captions_to_consider)
    captions = np.array(captions)
    negative_idxs = [i for i, c in enumerate(captions) if c in global_negative_tasks]
    for n in global_negative_tasks:
        tasks.discard(n)
    for task in tasks:
        idxs = [i for i, c in enumerate(captions_to_consider) if c == task]
        positive_idxs = [i for i, c in enumerate(captions) if c == task]
        neg_tasks = _get_negatives_for_task(task)
        task_negative_idxs = [i for i, c in enumerate(captions) if c in neg_tasks]
        task_scores = sim_matrix[idxs[0]]
        # Include only the current task and negative examples.
        included_scores = task_scores[
            positive_idxs + task_negative_idxs + negative_idxs
        ]
        included_captions = captions[positive_idxs + task_negative_idxs + negative_idxs]
        included_labels = [c == task for c in included_captions]
        task_to_auc[task] = roc_auc_score(included_labels, included_scores)
    return np.mean(list(task_to_auc.values()))


def get_strict_auc_per_task(
    sim_matrix,
    captions,
    labels,
    negative_tasks=None,
):
    """Get ROC-AUC with only a specific set of tasks as negative examples."""
    if negative_tasks is None:
        negative_tasks = NEGATIVE_TASKS
    task_to_auc = {}
    video_has_label = np.any(labels, axis=0)
    # Only average over caption types that match videos (leaves out negative captions).
    # Corresponds to the first dimension of sim_matrix.
    captions_to_consider = [
        caption for v, caption in enumerate(captions) if video_has_label[v]
    ]
    tasks = set(captions_to_consider)
    captions = np.array(captions)
    negative_idxs = [i for i, c in enumerate(captions) if c in negative_tasks]
    if len(negative_idxs) == 0:
        return -1
    for n in negative_tasks:
        tasks.discard(n)
    for task in tasks:
        idxs = [i for i, c in enumerate(captions_to_consider) if c == task]
        positive_idxs = [i for i, c in enumerate(captions) if c == task]
        task_scores = sim_matrix[idxs[0]]
        # Include only the current task and negative examples.
        included_scores = task_scores[positive_idxs + negative_idxs]
        included_captions = captions[positive_idxs + negative_idxs]
        included_labels = [c == task for c in included_captions]
        task_to_auc[task] = roc_auc_score(included_labels, included_scores)
    return np.mean(list(task_to_auc.values()))


def evaluate_auc(args, sim_matrix, captions, labels, datatype, metrics_dict=None):
    if metrics_dict is None:
        metrics_dict = {}
    auc = get_auc_per_task(sim_matrix, captions, labels)
    if args.success_data_only:
        # Strict AUC is not defined for datasets with only success examples.
        strict_auc = 0
    else:
        strict_auc = get_strict_auc_per_task(sim_matrix, captions, labels)
    if datatype in LABELED_NEGATIVE_TASKS:
        labeled_auc = get_labeled_auc_per_task(sim_matrix, captions, labels, datatype)
    else:
        labeled_auc = 0
    metrics_dict["auc"] = auc
    metrics_dict["labeled_auc"] = labeled_auc
    metrics_dict["strict_auc"] = strict_auc
    return metrics_dict


def load_vlm_video_eval(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    sim_matrix = data["sim_matrix"]
    if len(sim_matrix.shape) > 2:
        sim_matrix = sim_matrix[:, :, -1]
    captions = data["captions"]
    batched_video_ids = data["video_ids"]
    if "batch" in data:
        batch = data["batch"]
    video_ids = []
    for batch in batched_video_ids:
        video_ids.extend(batch)

    groups = collections.defaultdict(list)
    for i, video in enumerate(video_ids):
        group = "_".join(video.split("_")[:5])
        groups[group].append(i)
    result = (sim_matrix, captions, video_ids, groups)
    if "batch" in data:
        result = result + (batch,)
    return result


def get_vlm_accuracy(
    eval_path,
    cosine_scale=1.0,
    waypoint_only=False,
    condition=None,
    correct_condition=None,
    return_results=False,
    pred_captions=None,
    emb_dists=None,
    sim_weight=1.0,
    emb_weight=1.0,
    verbose=True,
    silent=False,
):
    eval_result = load_vlm_video_eval(eval_path)
    acc = compute_vlm_accuracy(
        *eval_result,
        cosine_scale=cosine_scale,
        waypoint_only=waypoint_only,
        condition=condition,
        correct_condition=correct_condition,
        pred_captions=pred_captions,
        emb_dists=emb_dists,
        sim_weight=sim_weight,
        emb_weight=emb_weight,
        verbose=verbose,
        silent=silent,
    )
    if return_results:
        return eval_result
    return acc


def compute_vlm_accuracy(
    sim_matrix,
    captions,
    video_ids,
    groups,
    *args,
    cosine_scale=1.0,
    waypoint_only=False,
    condition=None,
    correct_condition=None,
    pred_captions=None,
    emb_dists=None,
    sim_weight=1.0,
    emb_weight=1.0,
    verbose=True,
    silent=False,
):
    n_correct = 0
    n_incorrect = 0
    if correct_condition is None:
        correct_condition = lambda x: "waypoint1" in x
    for group in groups:
        idxs = groups[group]
        sims = sim_matrix[idxs, :]
        sims = sims[:, idxs]
        vids = []
        caps = []
        for idx in idxs:
            caps.append(captions[idx])
            vids.append(video_ids[idx])
        caption_idx = np.argmax([c != "No caption" for c in caps])
        if verbose:
            print(caps[caption_idx])
        scores = sims[caption_idx] * sim_weight / cosine_scale
        if pred_captions:
            pred_caps = [pred_captions[vid] for vid in vids]
            for v, caption in enumerate(pred_caps):
                if caption[0] == "perform failed grasp":
                    scores[v] = -1
        if emb_dists:
            pred_dists = [emb_dists[vid] for vid in vids]
            # Cosine distance: higher is more similar!
            for v, dist in enumerate(pred_dists):
                if emb_weight == "prob":
                    print(vids[v])
                    print("score before:", scores[v])
                    print(
                        "emb score to add:",
                        dist["emb_dist"][0],
                        dist["prob"][0],
                        dist["emb_dist"][0] * dist["prob"][0],
                    )
                    scores[v] += dist["emb_dist"][0] * dist["prob"][0]
                else:
                    scores[v] += dist["emb_dist"][0] * emb_weight
        video_to_scores = dict(zip(vids, scores))
        if waypoint_only:
            video_to_scores = {
                k: v for k, v in video_to_scores.items() if "waypoint" in k
            }
        if condition is not None:
            video_to_scores = {k: v for k, v in video_to_scores.items() if condition(k)}
        if verbose:
            print(group)
            for k, v in sorted(
                video_to_scores.items(), key=operator.itemgetter(1), reverse=True
            ):
                print(k, ":", v)
        max_kv = max(video_to_scores.items(), key=operator.itemgetter(1))
        if correct_condition(max_kv[0]):  # 'waypoint1' in max_kv[0]:
            n_correct += 1
        else:
            n_incorrect += 1
    if not silent:
        print(f"{n_correct} / {n_correct + n_incorrect} correct")
    return n_correct / (n_correct + n_incorrect)
