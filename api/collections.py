"""
Collections API (COL-1) - private folders of saved looks and saved outfits.

Thin routes. Real work lives in pipelines/collections_pipeline.py.

GET    /api/collections[?contains_type=look|outfit&contains_id=<uuid>]
         -> {"collections": [{collection_id, name, item_count, cover_image_url,
                              created_at, updated_at, contains?}]}  newest-updated first
POST   /api/collections                       {"name"}  -> 201 collection
PATCH  /api/collections/{collection_id}       {"name"}  -> collection
DELETE /api/collections/{collection_id}                 -> {collection_id, deleted}
GET    /api/collections/{collection_id}/entries?limit=50&offset=0
         -> {"entries": [{entry_type, added_at, look? | outfit?}], limit, offset}
POST   /api/collections/{collection_id}/entries  {"entry_type": "look"|"outfit", "entry_id"}
         -> 201 added / 200 already in
DELETE /api/collections/{collection_id}/entries/{entry_type}/{entry_id}

Errors: 400 COLLECTION_NAME_INVALID, COLLECTION_LIMIT_REACHED, INVALID_ENTRY_TYPE,
        INVALID_QUERY | 404 COLLECTION_NOT_FOUND, LOOK_NOT_FOUND, OUTFIT_NOT_FOUND,
        COLLECTION_ENTRY_NOT_FOUND | 409 COLLECTION_NAME_TAKEN
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.collections_pipeline import (
    list_collections, create_collection, rename_collection, delete_collection,
    list_entries, add_entry, remove_entry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])


class CollectionNameRequest(BaseModel):
    name: str | None = None


class AddEntryRequest(BaseModel):
    entry_type: str
    entry_id: UUID


@router.get("/health")
def health():
    return {"service": "collections", "version": "1.0.0", "status": "healthy"}


@router.get("")
def get_collections(
    contains_type: str | None = Query(None),
    contains_id: UUID | None = Query(None),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_collections(user_id, contains_type, contains_id))


@router.post("")
def post_collection(
    body: CollectionNameRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    response.status_code = 201
    return SuccessResponse(data=create_collection(user_id, body.name))


@router.patch("/{collection_id}")
def patch_collection(
    collection_id: UUID,
    body: CollectionNameRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=rename_collection(user_id, collection_id, body.name))


@router.delete("/{collection_id}")
def remove_collection(
    collection_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=delete_collection(user_id, collection_id))


@router.get("/{collection_id}/entries")
def get_entries(
    collection_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_entries(user_id, collection_id, limit, offset))


@router.post("/{collection_id}/entries")
def post_entry(
    collection_id: UUID,
    body: AddEntryRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    entry, created = add_entry(user_id, collection_id, body.entry_type, body.entry_id)
    response.status_code = 201 if created else 200
    return SuccessResponse(data=entry)


@router.delete("/{collection_id}/entries/{entry_type}/{entry_id}")
def delete_entry(
    collection_id: UUID,
    entry_type: str,
    entry_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=remove_entry(user_id, collection_id, entry_type, entry_id))