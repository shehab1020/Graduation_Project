import os
import random
from django.shortcuts import render
from .ai_model import evaluate_answer, calculate_level, calculate_score
from .model_engine import df
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status

from rest_framework.permissions import IsAuthenticated, AllowAny
from .roadmap_model import RoadmapModel

AVAILABLE_TRACKS = ["data_analysis", "frontend"]
AVAILABLE_LEVELS = ["beginner", "intermediate", "advanced"]


def select_questions(track, n_per_topic=2):
    questions_bank = (
        df[["question_id", "question",
            "difficulty", "topic", "track"]]
        .drop_duplicates(subset="question_id")
        .reset_index(drop=True)
    )

    track_qs = questions_bank[questions_bank["track"] == track]

    if len(track_qs) == 0:
        raise ValueError(f"No questions found for track: {track}")

    topics   = track_qs["topic"].unique().tolist()
    selected = []

    for topic in topics:
        topic_qs = track_qs[track_qs["topic"] == topic]
        sampled  = topic_qs.sample(min(n_per_topic, len(topic_qs)), random_state=None)
        for _, row in sampled.iterrows():
            selected.append(row.to_dict())

    random.shuffle(selected)
    return selected[:10]



def get_question_by_id(q_id):
    row = df[df["question_id"] == q_id]

    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "question": row["question"],
        "correct_answer": row["correct_answer"],
        "difficulty": row["difficulty"],
        "topic": row["topic"],
        "track": row["track"]
    }





@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_questions(request):
    track = request.query_params.get('track', None)

    if not track or (track not in AVAILABLE_TRACKS):
        return Response({'error':f'invalid track!! avaliable tracks: {AVAILABLE_TRACKS}'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        questions = select_questions(track=track)
        out = [
        {
            "question_id": q['question_id'],
            "question":q['question'],
            "difficulty":q['difficulty'],
            "topic":q['topic'],
            "track":q['track'],
        }
        for q in questions
        ]
        return Response({'track':track, 'questions': out}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': e}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['POST'])
def evaluate(request):
    if request.method == 'POST':
        answers = request.data.get('answers')
        if not answers:
            return Response({'error': 'answers are required!!'}, status=status.HTTP_400_BAD_REQUEST)
        result = []
        for answer in answers:
            q_id = answer['question_id']
            student_answer = answer['student_answer']

            out = get_question_by_id(q_id)
            correct_answer = out['correct_answer']
            result.append(evaluate_answer(correct_answer=correct_answer, student_answer=student_answer))
        score = calculate_score(result)
        level = calculate_level(score)
        return Response({'score':score, 'level':level, 'result':result}, status=status.HTTP_200_OK)




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(BASE_DIR, 'ml_model', 'roadmap_dataset.csv')

def get_model():
    return RoadmapModel(csv_path=dataset_dir)


@api_view(['GET'])
def generate_roadmap(request):
    model = get_model()
    track = request.query_params.get("track", None)
    level = request.query_params.get("level", None)
    if not track or not level:
        return Response({'Error':'track and level are required'}, status=status.HTTP_400_BAD_REQUEST)
    if track not in AVAILABLE_TRACKS:
        return Response({'Error':f'invalid track!! avaliable tracks: {AVAILABLE_TRACKS}'}, status=status.HTTP_400_BAD_REQUEST)
    if level not in AVAILABLE_LEVELS:
        return Response({'Error':f'invalid level!! avaliable tracks: {AVAILABLE_LEVELS}'}, status=status.HTTP_400_BAD_REQUEST)
    
    
    try:
        roadmap = model.generate(track=track, level=level)
        return Response({"data": roadmap})
    except ValueError as e:
        return Response({"Error": str(e)}, status=400)



@api_view(['GET'])
def get_options(request):
    model = get_model()
    return Response(model.get_options())

























# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# def evaluate(request):
#     data = request.data.get('answers', None)
#     if not data:
#         return Response({'Error': 'answers are required.'}, status=status.HTTP_400_BAD_REQUEST)
#     try:
#         total = []
#         track_for_overall_fuc = ''
#         for item in data:
#             question_id = item['question_id']
#             row = get_question_by_id(question_id)
#             question = row['question_text']
#             student_answer = item['student_answer']
#             difficulty = row['difficulty_level']
#             topic = row['topic_area']
#             track = row['track']
#             model_answer = row['model_answer']
            
#             results = grade_answer(question=question, model_answer=model_answer, student_answer=student_answer, difficulty=difficulty,track=track, topic=topic)

#             q_score = get_question_score(results['label'])
#             results['q_score'] = q_score
#             results["difficulty_level"] = difficulty
#             results["topic_area"] = topic
#             results["track"] = track
#             track_for_overall_fuc = track
#             total.append(results)
#         final = get_overall_result(total, track_for_overall_fuc)
#     except:
#         raise Exception('Evaluation Failed')

#     return Response(final)










































# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# def evaluate(reqeust):
#     data = reqeust.data
#     required = [ 'question_id' ,'question_text', 'student_answer', 'difficulty_level', 'topic_area', 'track']
#     for r in required:
#         if r not in data:
#             return Response({'error': 'invalid inputs'}, status=status.HTTP_400_BAD_REQUEST)
#     question_id = data['question_id']
#     question = data['question_text']
#     student_answer = data['student_answer']
#     difficulty = data['difficulty_level']
#     topic = data['topic_area']
#     track = data['track']
#     model_answer = get_question_by_id(question_id)
#     model_answer = model_answer['model_answer']

#     results = grade_answer(question=question, model_answer=model_answer, student_answer=student_answer, difficulty=difficulty,track=track, topic=topic)

#     q_score = get_question_score(results['label'])
#     results['q_score'] = q_score
#     return Response(results)












# [
#         {
#             "question_id": "FE016",
#             "question_text": "What is a CSS media query?",
#             "difficulty_level": "medium",
#             "topic_area": "CSS",
#             "track": "Front-End",
#              "q_score": 9.0,
#         },
#         {
#             "question_id": "FE002",
#             "question_text": "What is the difference between block and inline elements in HTML?",
#             "difficulty_level": "easy",
#             "topic_area": "HTML",
#             "track": "Front-End"
#         },
#         {
#             "question_id": "FE004",
#             "question_text": "What is semantic HTML?",
#             "difficulty_level": "medium",
#             "topic_area": "HTML",
#             "track": "Front-End"
#         },
# ]






# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# def submit_assessment(request):
#     data = request.data
#     answers = data.get("answers", [])

#     if not answers:
#         return Response({"error": "No answers provided"}, status=400)

#     results = []

#     for item in answers:
#         q_id = item.get("question_id")
#         student_answer = item.get("student_answer")

#         q_data = get_question_by_id(q_id)

#         if not q_data:
#             continue

#         result = grade_answer(
#             question=q_data["question_text"],
#             model_answer=q_data["model_answer"],
#             student_answer=student_answer,
#             difficulty=q_data["difficulty_level"],
#             topic=q_data["topic_area"],
#             track=q_data["track"]
#         )

#         q_score = get_question_score(result["label"])

#         results.append({
#             "topic": q_data["topic_area"],
#             "difficulty": q_data["difficulty_level"],
#             "score": q_score
#         })

#     #  حساب التوبيك
#     topic_scores = {}
#     difficulty_weights = {"easy": 1, "medium": 2, "hard": 3}

#     for r in results:
#         topic = r["topic"]
#         weight = difficulty_weights.get(r["difficulty"], 2)

#         if topic not in topic_scores:
#             topic_scores[topic] = {"score": 0, "max": 0}

#         topic_scores[topic]["score"] += r["score"] * weight
#         topic_scores[topic]["max"] += 10 * weight

#     #  overall
#     total_score = sum(v["score"] for v in topic_scores.values())
#     total_max = sum(v["max"] for v in topic_scores.values())

#     overall_pct = (total_score / total_max * 100) if total_max > 0 else 0

#     #  topics output
#     topics_out = {}

#     for topic, vals in topic_scores.items():
#         pct = (vals["score"] / vals["max"] * 100)
#         topics_out[topic] = {
#             "score": round(pct, 1),
#             "level": get_level(pct)
#         }

#     return Response({
#         "overall": {
#             "score": round(overall_pct, 1),
#             "level": get_level(overall_pct)
#         },
#         "topics": topics_out
#     })



