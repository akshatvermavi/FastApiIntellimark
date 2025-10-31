from fastapi import APIRouter, Form, HTTPException
import os
import pandas as pd
import secrets
from datetime import datetime
from typing import Optional

# Reuse the same token CSV path pattern as main.py
TOKEN_CSV_PATH = os.path.join(os.path.dirname(__file__), 'Token', 'user_token.csv')

router = APIRouter()


def _ensure_store() -> pd.DataFrame:
    os.makedirs(os.path.dirname(TOKEN_CSV_PATH), exist_ok=True)
    if not os.path.exists(TOKEN_CSV_PATH):
        df = pd.DataFrame(columns=['username', 'token', 'counter'])
        df.to_csv(TOKEN_CSV_PATH, index=False)
        return df
    try:
        df = pd.read_csv(TOKEN_CSV_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read token store: {e}')
    # Normalize and ensure required columns
    df.columns = [c.lower() for c in df.columns]
    for col in ['username', 'token', 'counter']:
        if col not in df.columns:
            df[col] = pd.Series(dtype=object)
    return df


def _save_store(df: pd.DataFrame) -> None:
    try:
        df.to_csv(TOKEN_CSV_PATH, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to persist token store: {e}')


@router.post('/generate-token')
async def generate_token(username: str = Form(...), token: Optional[str] = Form(None), counter: Optional[str] = Form(None)):
    """
    Create or refresh an access token for the given username.
    - Optional `token`: use a custom token (must be unique or owned by same username). If omitted, a random token is generated.
    - Optional `counter`: starting credits (non-negative integer). Defaults to 100.
    """
    uname = str(username).strip()
    if not uname:
        raise HTTPException(status_code=400, detail='username is required')

    # Parse counter
    if counter is None or str(counter).strip() == "":
        desired_counter = 100
    else:
        try:
            desired_counter = int(str(counter).strip())
            if desired_counter < 0:
                raise ValueError
        except Exception:
            raise HTTPException(status_code=400, detail='counter must be a non-negative integer')

    df = _ensure_store()

    # Decide the token value
    requested = None if token is None else str(token).strip()
    existing_tokens = set(df['token'].astype(str)) if 'token' in df.columns else set()

    if requested:
        # If token already exists for another user, reject
        if requested in existing_tokens:
            owners = df[df['token'].astype(str) == requested]['username'].astype(str).unique().tolist()
            if len(owners) > 0 and owners[0] != uname:
                raise HTTPException(status_code=400, detail='token already in use by another user')
        final_token = requested
    else:
        final_token = secrets.token_urlsafe(24)
        while final_token in existing_tokens:
            final_token = secrets.token_urlsafe(24)

    # Update or append
    if (df['username'].astype(str) == uname).any():
        idx = df.index[df['username'].astype(str) == uname][0]
        df.at[idx, 'token'] = final_token
        df.at[idx, 'counter'] = desired_counter
        action = 'refreshed'
    else:
        new_row = pd.DataFrame({'username': [uname], 'token': [final_token], 'counter': [desired_counter]})
        df = pd.concat([df, new_row], ignore_index=True)
        action = 'created'

    _save_store(df)

    return {
        'username': uname,
        'token': final_token,
        'counter': desired_counter,
        'status': action,
        'message': f'token {action} for username; initial counter={desired_counter}'
    }
