
import os
import json
import glob
import math
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _clean(obj):
    """Remove NaN and convert to clean Python types."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    return obj


def _videos(lesson: dict) -> dict:
    """
    Return a structured videos dict with top 3 options.
    video_1 → curated/hand-picked
    video_2 → top result from best channel for this track
    video_3 → alternative channel / different teaching style
    """
    return {
        "video_1": lesson.get("video_1"),
        "video_2": lesson.get("video_2"),
        "video_3": lesson.get("video_3"),
    }


def _build_lesson(lesson: dict, topic_name: str) -> dict:
    return {
        "lesson_id":        lesson.get("lesson_id"),
        "topic":            topic_name,
        "subtopic":         lesson.get("subtopic"),
        "level":            lesson.get("level"),
        "content_type":     lesson.get("content_type"),
        "description":      lesson.get("description_en"),
        "duration_minutes": lesson.get("duration_minutes"),
        "project_idea":     lesson.get("project_idea"),
        "resources": {
            "videos":   _videos(lesson),
            "article":  lesson.get("article_url"),
        },
    }


# ─────────────────────────────────────────────
# Core Model
# ─────────────────────────────────────────────

class RoadmapModel:
    """
    Loads the roadmap CSV and generates structured roadmaps.

    Parameters
    ----------
    csv_path : str
        Path to roadmap_master_dataset_updated.csv
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self._validate_columns()

    # ── public ──────────────────────────────

    def get_options(self) -> dict:
        """Return all available tracks and levels."""
        return {
            "tracks": sorted(self.df["track"].dropna().unique().tolist()),
            "levels": sorted(
                self.df["level"].dropna().unique().tolist(),
                key=lambda x: ["beginner", "intermediate", "advanced"].index(x.lower())
                if x.lower() in ["beginner", "intermediate", "advanced"] else 99
            ),
            "topics_by_track": {
                track: sorted(
                    self.df[self.df["track"] == track]["topic"].dropna().unique().tolist()
                )
                for track in self.df["track"].dropna().unique()
            },
        }

    def generate(
        self,
        track: str,
        level: str,
        topic: Optional[str] = None,
        page: int = 1,
        page_size: int = 0,      # 0 = no pagination
    ) -> dict:
        """
        Generate a roadmap for a given track + level.

        Parameters
        ----------
        track     : e.g. "Frontend"
        level     : e.g. "beginner"
        topic     : optional filter e.g. "HTML"
        page      : page number (1-indexed), used only if page_size > 0
        page_size : number of topics per page, 0 = return all
        """
        df = self._filter(track, level, topic)
        if df.empty:
            raise ValueError(
                f"No lessons found for track='{track}', level='{level}'"
                + (f", topic='{topic}'" if topic else "")
            )

        topics_ordered, topics_map = self._group_by_topic(df)

        # ── build roadmap steps ──
        roadmap = []
        for step_num, (topic_id, topic_name) in enumerate(topics_ordered, start=1):
            lessons_raw = topics_map[(topic_id, topic_name)]
            roadmap.append({
                "step":         step_num,
                "topic_id":     topic_id,
                "main_topic":   topic_name,
                "total_lessons": len(lessons_raw),
                "lessons": [
                    _build_lesson(_clean(l), topic_name)
                    for l in lessons_raw
                ],
            })

        # ── optional pagination ──
        total_topics = len(roadmap)
        pagination   = None
        if page_size > 0:
            total_pages = math.ceil(total_topics / page_size)
            start       = (page - 1) * page_size
            roadmap     = roadmap[start: start + page_size]
            pagination  = {
                "page":         page,
                "page_size":    page_size,
                "total_topics": total_topics,
                "total_pages":  total_pages,
                "has_next":     page < total_pages,
                "has_prev":     page > 1,
            }

        result = {
            "track":        track,
            "level":        level,
            "total_topics": total_topics,
            "total_lessons": sum(t["total_lessons"] for t in roadmap),
            "roadmap":      roadmap,
        }
        if topic:
            result["topic_filter"] = topic
        if pagination:
            result["pagination"] = pagination

        return result

    def get_lesson(self, lesson_id: str) -> dict:
        """Fetch a single lesson by its ID."""
        row = self.df[self.df["lesson_id"] == lesson_id]
        if row.empty:
            raise ValueError(f"Lesson '{lesson_id}' not found.")
        lesson = _clean(row.iloc[0].to_dict())
        return _build_lesson(lesson, lesson.get("topic", ""))






    # ── private ─────────────────────────────

    def _validate_columns(self):
        required = {"track", "level", "topic", "subtopic", "lesson_id",
                    "video_1", "video_2", "video_3"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

    def _filter(self, track: str, level: str, topic: Optional[str]) -> pd.DataFrame:
        mask = (
            (self.df["track"].str.lower() == track.lower()) &
            (self.df["level"].str.lower() == level.lower())
        )
        if topic:
            mask &= (self.df["topic"].str.lower() == topic.lower())
        return self.df[mask].copy()

    @staticmethod
    def _group_by_topic(df: pd.DataFrame):
        topics_seen = []
        topics_map  = {}
        for lesson in df.to_dict(orient="records"):
            key = (lesson["topic_id"], lesson["topic"])
            if key not in topics_map:
                topics_map[key] = []
                topics_seen.append(key)
            topics_map[key].append(lesson)
        return topics_seen, topics_map
    

"""---
## 📚 API Reference

| Method | Parameters | Description |
|---|---|---|
| `get_options()` | — | جميع الـ tracks والـ levels المتاحة |
| `generate(track, level, topic?, page?, page_size?)` | track, level إجباريان | توليد roadmap كاملة أو مفلترة |
| `get_lesson(lesson_id)` | lesson_id | جلب lesson واحدة بالـ ID |

### مثال Django / Flask
كود الـ integration موجود في أسفل الملف الأصلي `roadmap_model.py`.

---
*RoadmapModel v2 — Generated Notebook*
"""