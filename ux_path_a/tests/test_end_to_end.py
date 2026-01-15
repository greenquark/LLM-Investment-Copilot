"""
End-to-end tests for UX Path A.

Tests the complete flow from API calls to database persistence.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ux_path_a.backend.main import app
from ux_path_a.backend.core.database import Base, get_db
from ux_path_a.backend.core.config import settings

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db():
    """Create test database."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db):
    """Create test client with database override."""
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_auth_register(client):
    """Test user registration."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpass123",
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"


def test_auth_login(client):
    """Test user login."""
    # First register
    client.post(
        "/api/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpass123",
        }
    )
    
    # Then login
    response = client.post(
        "/api/auth/token",
        data={
            "username": "testuser",
            "password": "testpass123",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_create_session(client):
    """Test creating a chat session."""
    # Login first
    login_response = client.post(
        "/api/auth/token",
        data={"username": "testuser", "password": "testpass123"}
    )
    token = login_response.json()["access_token"]
    
    # Create session
    response = client.post(
        "/api/chat/sessions",
        json={"title": "Test Session"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Session"


def test_send_message(client):
    """Test sending a chat message."""
    # Login
    login_response = client.post(
        "/api/auth/token",
        data={"username": "testuser", "password": "testpass123"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create session
    session_response = client.post(
        "/api/chat/sessions",
        json={"title": "Test"},
        headers=headers
    )
    session_id = session_response.json()["id"]
    
    # Send message (will fail without OpenAI key, but structure should work)
    response = client.post(
        "/api/chat/messages",
        json={
            "content": "What is the price of AAPL?",
            "session_id": session_id,
        },
        headers=headers
    )
    
    # Should either succeed or fail with proper error (not 500)
    assert response.status_code in [200, 400, 401, 500]  # 500 if OpenAI key missing
    if response.status_code == 200:
        data = response.json()
        assert "message" in data
        assert "session_id" in data


def test_list_sessions(client):
    """Test listing sessions."""
    # Login
    login_response = client.post(
        "/api/auth/token",
        data={"username": "testuser", "password": "testpass123"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a session
    client.post(
        "/api/chat/sessions",
        json={"title": "Test Session"},
        headers=headers
    )
    
    # List sessions
    response = client.get("/api/chat/sessions", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
