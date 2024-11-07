import json
import os
import numpy as np
from typing import Dict, Any, List
import random

def determine_thresholds(input_folder: str) -> List[float]:
    all_scores = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_scores.append(data['total_segment_score'])
    
    all_scores = np.array(all_scores)
    thresholds = [
        np.percentile(all_scores, 90),  # Outstanding
        np.percentile(all_scores, 70),  # Exceptional
        np.percentile(all_scores, 50),  # Strong
        np.percentile(all_scores, 30),  # Promising
    ]
    return thresholds


def evaluate_performance(total_segment_score: float, thresholds: List[float]) -> str:
    outstanding = [
        "delivered an outstanding performance that left the audience in awe",
        "gave a breathtaking performance that will be remembered for years to come",
        "showcased world-class skill and artistry, setting a new standard for excellence",
        "delivered a flawless routine that captivated everyone in attendance",
        "executed a performance of the highest caliber, earning thunderous applause",
        "demonstrated unparalleled mastery of both technical elements and artistic expression",
        "mesmerized the crowd with an extraordinary display of skill and grace",
        "delivered a tour de force performance that pushed the boundaries of the sport",
        "showcased perfection on ice, leaving spectators and judges alike spellbound",
        "gave a performance for the ages, setting a new benchmark in figure skating"
    ]
    
    exceptional = [
        "showcased exceptional skill and artistry",
        "delivered a remarkable performance that impressed both judges and spectators",
        "demonstrated masterful technique and emotional expression",
        "executed a near-perfect routine with grace and precision",
        "displayed outstanding control and fluidity throughout the program",
        "delivered a performance of exceptional quality, highlighting years of dedication",
        "exhibited remarkable poise and technical prowess in a stellar routine",
        "showcased a harmonious blend of athletic skill and artistic interpretation",
        "delivered a captivating performance that stood out for its excellence",
        "demonstrated exceptional talent with a nearly flawless execution"
    ]
    
    strong = [
        "demonstrated strong technical ability and artistic expression",
        "showed great skill and poise throughout the performance",
        "delivered a solid routine with several standout moments",
        "exhibited confidence and control in a well-executed program",
        "showcased a strong command of difficult elements and artistic interpretation",
        "delivered a performance marked by technical precision and expressive skating",
        "demonstrated a high level of skill with a few minor imperfections",
        "showed impressive control and artistry in a well-rounded performance",
        "executed a strong program with moments of brilliance",
        "displayed notable talent and potential in a commendable routine"
    ]
    
    promising = [
        "showed promising talent with room for growth",
        "delivered a commendable performance with potential for improvement",
        "displayed good fundamentals and some impressive elements",
        "showed determination and spirit in a developing routine",
        "exhibited potential with several well-executed elements",
        "demonstrated developing skills with moments of notable execution",
        "showed promise with a mix of well-performed elements and areas for improvement",
        "delivered a spirited performance showcasing emerging talent",
        "displayed good effort and technique with room for refinement",
        "showed glimpses of future excellence in a developing program"
    ]
    
    challenging = [
        "faced some challenges but showed potential for improvement",
        "encountered difficulties but demonstrated resilience throughout the performance",
        "showed moments of promise despite facing obstacles",
        "displayed courage in completing the routine despite setbacks",
        "struggled with some elements but showed determination to finish strong",
        "faced technical challenges yet demonstrated a fighting spirit",
        "experienced some setbacks but showed commitment to the performance",
        "had a difficult outing but displayed character and perseverance",
        "struggled with consistency but showed flashes of potential",
        "faced adversity in the performance but demonstrated a willingness to learn and improve"
    ]

    if total_segment_score >= thresholds[0]:
        return random.choice(outstanding)
    elif total_segment_score >= thresholds[1]:
        return random.choice(exceptional)
    elif total_segment_score >= thresholds[2]:
        return random.choice(strong)
    elif total_segment_score >= thresholds[3]:
        return random.choice(promising)
    else:
        return random.choice(challenging)

def generate_simple_caption(label: Dict[str, Any], thresholds: List[float]) -> str:
    category = "free skate" if "fs" in label['competition'].lower() else "short program"
    gender = "women's" if "women" in label['competition'].lower() else "men's"
    
    performance_evaluation = evaluate_performance(label['total_segment_score'], thresholds)
    
    caption = f"In this {gender} {category} competition, the skater {performance_evaluation}."
    
    return caption

def process_json_files(input_folder: str):
    thresholds = determine_thresholds(input_folder)
    print(f"Determined thresholds: {thresholds}")

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            simple_caption = generate_simple_caption(data, thresholds)
            data['simple_caption'] = simple_caption
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Processed {filename}")

    print("All files processed successfully.")

# Set your input folder path
input_folder = "/data1/1/code/datasets/fs/finefs/annotation"

# Run the batch processing
process_json_files(input_folder)