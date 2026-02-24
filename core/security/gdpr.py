from datetime import datetime, timezone
from typing import Any, Dict

from ..db import db, fetch_all


class GDPRCompliance:
    """GDPR compliance utilities."""
    
    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all data for a user (Right to Erasure - Article 17)."""
        deleted: Dict[str, int] = {}
        
        async with db.connect() as conn:
            # Delete sessions
            cur = await conn.execute("DELETE FROM sessions WHERE id = ?", (user_id,))
            deleted["sessions"] = int(getattr(cur, "rowcount", 0) or 0)
            
            # Delete tasks
            cur = await conn.execute("DELETE FROM tasks WHERE session_id = ?", (user_id,))
            deleted["tasks"] = int(getattr(cur, "rowcount", 0) or 0)
            
            # Delete episodes
            cur = await conn.execute("DELETE FROM episodes WHERE session_id = ?", (user_id,))
            deleted["episodes"] = int(getattr(cur, "rowcount", 0) or 0)
            
            # Delete preferences
            cur = await conn.execute("DELETE FROM preferences WHERE session_id = ?", (user_id,))
            deleted["preferences"] = int(getattr(cur, "rowcount", 0) or 0)
            
            # Delete working memory
            cur = await conn.execute("DELETE FROM working_memory WHERE session_id = ?", (user_id,))
            deleted["working_memory"] = int(getattr(cur, "rowcount", 0) or 0)
            
            # Delete session snapshots
            cur = await conn.execute("DELETE FROM session_memory WHERE session_id = ?", (user_id,))
            deleted["session_snapshots"] = int(getattr(cur, "rowcount", 0) or 0)
            
            # Anonymize audit logs (keep for compliance, remove user data)
            cur = await conn.execute(
                "UPDATE audit_log SET user_id = '[DELETED]' WHERE user_id = ?",
                (user_id,)
            )
            deleted["audit_anonymized"] = int(getattr(cur, "rowcount", 0) or 0)
            
            await conn.commit()
        
        return {"ok": True, "deleted": deleted}
    
    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data (Right to Portability - Article 20)."""
        export: Dict[str, Any] = {
            "user_id": user_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        
        async with db.connect() as conn:
            # Sessions
            rows = await fetch_all(
                conn,
                "SELECT * FROM sessions WHERE id = ?",
                (user_id,)
            )
            export["sessions"] = [dict(r) for r in rows]
            
            # Episodes
            rows = await fetch_all(
                conn,
                "SELECT * FROM episodes WHERE session_id = ?",
                (user_id,)
            )
            export["episodes"] = [dict(r) for r in rows]
            
            # Preferences
            rows = await fetch_all(
                conn,
                "SELECT * FROM preferences WHERE session_id = ?",
                (user_id,)
            )
            export["preferences"] = [dict(r) for r in rows]
        
        return export
    
    async def anonymize_data(self, user_id: str) -> Dict[str, Any]:
        """Anonymize user data while keeping for analytics."""
        async with db.connect() as conn:
            # Anonymize sessions
            await conn.execute(
                "UPDATE sessions SET title = '[ANONYMIZED]' WHERE id = ?",
                (user_id,)
            )
            
            # Anonymize episodes
            await conn.execute(
                "UPDATE episodes SET title = '[ANONYMIZED]', summary = '[ANONYMIZED]' WHERE session_id = ?",
                (user_id,)
            )
            
            # Anonymize working memory
            await conn.execute(
                "UPDATE working_memory SET content = '[ANONYMIZED]' WHERE session_id = ?",
                (user_id,)
            )
            
            await conn.commit()
        
        return {"ok": True, "anonymized": user_id}


gdpr = GDPRCompliance()
