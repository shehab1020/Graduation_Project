from django.shortcuts import render
from .ai_model import select_questions, get_level, get_question_score, grade_answer, get_overall_result, AVAILABLE_TRACKS
from .model_engine import scaler, xgb, bert_model, df
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status

from rest_framework.permissions import IsAuthenticated, AllowAny



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_questions(request):
    track = request.query_params.get('track', None)

    if not track or (track not in AVAILABLE_TRACKS):
        return Response({'error':f'invalid track!! avaliable tracks: {AVAILABLE_TRACKS}'}, status=status.HTTP_400_BAD_REQUEST)
    
    questions = select_questions(track=track)
    try:
        out = [
        {
            "question_id": q['question_id'],
            "question_text":q['question_text'],
            "difficulty_level":q['difficulty_level'],
            "topic_area":q['topic_area'],
            "track":q['track'],
        }
        for q in questions
        ]
        return Response({'track':track, 'questions': out}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': e}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




def get_question_by_id(q_id):
    row = df[df["question_id"] == q_id]

    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "question_text": row["question_text"],
        "model_answer": row["model_answer"],
        "difficulty_level": row["difficulty_level"],
        "topic_area": row["topic_area"],
        "track": row["track"]
    }


# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# def evaluate(request):
#     data = request.data.get('answers')
#     if not data:
#         return Response({"error": "answers required"}, status=400)

#     total = []

#     for item in data:
#         question_id = item['question_id']
#         row = get_question_by_id(question_id)

#         if not row:
#             return Response({"error": f"Invalid question_id: {question_id}"}, status=400)

#         results = grade_answer(
#             question=row['question_text'],
#             model_answer=row['model_answer'],
#             student_answer=item['student_answer'],
#             difficulty=row['difficulty_level'],
#             topic=row['topic_area'],
#             track=row['track']
#         )

#         results['q_score'] = get_question_score(results['label'])
#         results['difficulty_level'] = row['difficulty_level']
#         results['topic_area'] = row['topic_area']
#         results['track'] = row['track']

#         total.append(results)

#     track = total[0]['track'] if total else 'Unknown'

#     final = get_overall_result(total, track)

#     return Response(final)





@api_view(['POST'])
@permission_classes([IsAuthenticated])
def evaluate(request):
    data = request.data.get('answers', None)
    if not data:
        return Response({'Error': 'answers are required.'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        total = []
        track_for_overall_fuc = ''
        for item in data:
            question_id = item['question_id']
            row = get_question_by_id(question_id)
            question = row['question_text']
            student_answer = item['student_answer']
            difficulty = row['difficulty_level']
            topic = row['topic_area']
            track = row['track']
            model_answer = row['model_answer']
            
            results = grade_answer(question=question, model_answer=model_answer, student_answer=student_answer, difficulty=difficulty,track=track, topic=topic)

            q_score = get_question_score(results['label'])
            results['q_score'] = q_score
            results["difficulty_level"] = difficulty
            results["topic_area"] = topic
            results["track"] = track
            track_for_overall_fuc = track
            total.append(results)
        final = get_overall_result(total, track_for_overall_fuc)
    except:
        raise Exception('Evaluation Failed')

    return Response(final)










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



