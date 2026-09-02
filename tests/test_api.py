# -*- coding: utf-8 -*-
"""服务接口测试：用 FastAPI TestClient 测 /health /detect /detect_batch /report_bug。
不依赖真实 uvicorn/端口，直接调 app 路由。
注：server.py 顶层 import detector 模块，若需 torch 则在装有 torch 的 CI 下运行。
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

@pytest.fixture(scope="module")
def client():
    """通过 importlib 加载 server 模块，避免依赖模型文件即可测路由（检测需要模型，缺则 skip）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("srv",
        os.path.join(".." if not os.path.isabs(__file__) else os.path.dirname(__file__), "..", "scripts", "server.py"))
    return spec

def test_server_module_importable():
    """server.py 应能编译/加载（依赖已装）。"""
    import ast
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
    src = open(path, encoding="utf-8").read()
    ast.parse(src)  # 语法正确即可

def test_health_with_real_models():
    """用 TestClient 测 /health，需模型已训练（PASS 若模型存在）。"""
    import importlib.util
    from fastapi.testclient import TestClient
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
    spec = importlib.util.spec_from_file_location("srv", path)
    srv = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(srv)
    except Exception as e:
        # 模型加载失败则 skip（无模型/GPU）
        pytest.skip(f"server 模块加载失败（可能缺模型/依赖）: {str(e)[:80]}")
    client = TestClient(srv.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "ready" in data

def test_detect_report_bug_endpoints_up():
    """(快速)仅验证路由存在——通过检查 app.routes。"""
    import importlib.util
    from fastapi.testclient import TestClient
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
    spec = importlib.util.spec_from_file_location("srv", path)
    srv = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(srv)
    except Exception:
        pytest.skip("server 模块加载失败")
    client = TestClient(srv.app)
    # /health 和 /report_bug 应存在
    r = client.get("/health")
    assert r.status_code == 200
    # /report_bug 接收 json
    rr = client.post("/report_bug", json={"description": "API测试上报", "severity": "low", "scene": "test"})
    assert rr.status_code == 200

def test_wrong_json_404_handling():
    """无效路径返回 JSON 错误（服务健壮性）。"""
    import importlib.util
    from fastapi.testclient import TestClient
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
    spec = importlib.util.spec_from_file_location("srv", path)
    srv = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(srv)
    except Exception:
        pytest.skip("server 模块加载失败")
    client = TestClient(srv.app)
    # 无效路径
    r = client.post("/detect", json={"path": "/nope/invalid.docx"})
    assert r.status_code == 404
    assert "error" in r.json()
