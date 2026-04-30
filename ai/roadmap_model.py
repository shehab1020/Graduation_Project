
import os
import json
import glob
import pandas as pd


def _sanitize(obj):
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj


class RoadmapModel:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

    def get_options(self) -> dict:
        return {
            "tracks": self.df["track"].unique().tolist(),
            "levels": self.df["level"].unique().tolist(),
        }

    def generate(self, track: str, level: str) -> dict:
        mask = (
            (self.df["track"].str.lower() == track.lower()) &
            (self.df["level"].str.lower() == level.lower())
        )
        lessons = _sanitize(self.df[mask].to_dict(orient="records"))

        topics_seen = []
        topics_map  = {}
        for lesson in lessons:
            key = (lesson["topic_id"], lesson["topic"])
            if key not in topics_map:
                topics_map[key] = []
                topics_seen.append(key)
            topics_map[key].append(lesson)

        roadmap = []
        for step_num, (topic_id, topic_name) in enumerate(topics_seen, start=1):
            roadmap.append({
                "step":       step_num,
                "topic_id":   topic_id,
                "main_topic": topic_name,
                "lessons": [
                    {
                        "lesson_id":    l["lesson_id"],
                        "topic":        topic_name,
                        "subtopic":     l["subtopic"],
                        "content_type": l["content_type"],
                        "description":  l["description_en"],
                        "resources": {
                            "video":   l["youtube_link"],
                            "article": l["article_url"],
                        }
                    }
                    for l in topics_map[(topic_id, topic_name)]
                ]
            })

        return {
            "track":   track,
            "level":   level,
            "roadmap": roadmap,
        }

if __name__ == "__main__":
    csv_files = glob.glob("/content/*.csv")
    CSV_PATH  = csv_files[0]
    print(f"Using: {CSV_PATH}")

    model   = RoadmapModel(csv_path=CSV_PATH)
    roadmap = model.generate(track="Frontend", level="beginner")

    print(json.dumps(roadmap, indent=2, ensure_ascii=False))