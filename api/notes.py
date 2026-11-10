"""
Fit Notes API (NOTE-1, 2026-11-10). Work lives in pipelines/notes_pipeline.py.

GET    /api/notes/row                     -> {mine: note|null, items: [note]}  (people I follow, newest first, cap 100)
GET    /api/notes/{id}                    -> note | 404 NOTE_NOT_FOUND | 410 NOTE_EXPIRED
POST   /api/notes                         {text?, text_bg?, source_type?, source_id?, item_ids?} -> 201 note
DELETE /api/notes/{id}                    -> {deleted: true}           (author only)
PUT    /api/notes/{id}/reaction           {reaction} -> {my_reaction}
DELETE /api/notes/{id}/reaction           -> {my_reaction: null}
GET    /api/notes/{id}/reactions?type=&cursor=&limit=   (author only, live only)
         -> {totals: {love, fire, clean, bold, wear, all}, items: [{user, reaction, reacted_at, ...user}], next_cursor}
POST   /api/notes/{id}/report             {reason, note?} -> {reported: true}

note = {note_id, kind, text, text_bg, category, pieces[], author{user_id, display_name,
        handle, avatar_url_small, is_me}, created_at, expires_at, my_reaction,
        reaction_summary{total, top_actors[]} (author only)}

Errors: 400 TEXT_TOO_LONG | TEXT_REQUIRED | TEXT_BG_INVALID | SOURCE_INVALID | PLAN_NOT_TODAY |
        OUTFIT_EMPTY | INVALID_REACTION | CANNOT_REACT_OWN | CANNOT_REPORT_SELF | INVALID_CURSOR
        403 HANDLE_REQUIRED | NOT_ALLOWED   404 NOTE_NOT_FOUND   410 NOTE_EXPIRED
        409 NOTE_CONFLICT   429 RATE_LIMITED (+ Retry-After)
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.notes_pipeline import (
    notes_row, get_note, create_note, delete_note, react, unreact, list_reactions, report_note,
)

router = APIRouter(prefix="/notes", tags=["Fit Notes"])


class NewNoteRequest(BaseModel):
    text: str | None = None
    text_bg: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    item_ids: list[str] | None = None


class ReactionRequest(BaseModel):
    reaction: str


class ReportRequest(BaseModel):
    reason: str
    note: str | None = None


@router.get("/health")
def health():
    return {"service": "notes", "version": "1.0.0", "status": "healthy"}


@router.get("/row")
def get_row(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=notes_row(user_id))


@router.post("")
def post_note(body: NewNoteRequest, response: Response,
              user_id: UUID = Depends(get_current_user_id)):
    response.status_code = 201
    return SuccessResponse(data=create_note(user_id, body.model_dump()))


@router.get("/{note_id}")
def read_note(note_id: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_note(note_id, user_id))


@router.delete("/{note_id}")
def remove_note(note_id: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=delete_note(note_id, user_id))


@router.put("/{note_id}/reaction")
def put_reaction(note_id: str, body: ReactionRequest,
                 user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=react(note_id, user_id, body.reaction))


@router.delete("/{note_id}/reaction")
def delete_reaction(note_id: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=unreact(note_id, user_id))


@router.get("/{note_id}/reactions")
def get_reactions(note_id: str, type: str | None = Query(None), cursor: str | None = Query(None),
                  limit: int = Query(30, ge=1, le=50),
                  user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=list_reactions(note_id, user_id, type, cursor, limit))


@router.post("/{note_id}/report")
def post_report(note_id: str, body: ReportRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=report_note(note_id, user_id, body.reason, body.note))