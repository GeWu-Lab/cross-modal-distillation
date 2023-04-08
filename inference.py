import time
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from utils import AverageMeter
from scipy.ndimage import zoom
import numpy as np


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs, k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def get_similarity(video_feature,audio_feature):
    norm_video = F.normalize(video_feature, dim = 1)
    norm_audio = F.normalize(audio_feature, dim = 1)

    similarity = torch.mm(norm_video,norm_audio.T)


    return similarity


def inference(data_loader, model, result_path, class_names, no_average, output_topk):
    print('inference')

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}
    

    end_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            targets = batch['target'].cuda()
            inputs = batch['clip'].cuda()

            data_time.update(time.time() - end_time)

            video_name = batch['video_name']

            #TODO:change
            _,_,outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1).cpu()


            for j in range(outputs.size(0)):
                results['results'][video_name[j]].append({
                    'output': outputs[j]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
            # print(inference_results['results'][video_id])

    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    # print(inference_results)

    with result_path.open('w') as f:
        json.dump(inference_results, f)
