import json
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

# These tests check the UI and flow logic with a FAKE Gemini client (MagicMock), so they
# run offline and cost nothing. They do NOT tell you whether the real model answers well —
# that needs real API calls (eval).

fake = {"category":"商品","suggested_reply":"測試回覆","source":"FAQ013","confidence":"高",
        "handling":"可直接採用","confidence_reason":"知識庫有明確對應"}
fake_nokb = {"category":"其他","suggested_reply":"目前知識庫沒有對應資訊，建議轉人工確認","source":None,
        "confidence":"低","handling":"請轉人工處理","confidence_reason":"無對應"}
fake_bad_source = dict(fake, source="FAQ999")

def make_client(text):
    """Build a fake genai.Client whose generate_content returns `text`."""
    c = MagicMock()
    c.models.generate_content.return_value = MagicMock(text=text)
    return c

def payload(d):
    return json.dumps(d, ensure_ascii=False)

def click_analyze(at):
    [b for b in at.button if b.label=="🔍 分析問題"][0].click().run()

at = AppTest.from_file("app.py", default_timeout=30)
at.secrets["GEMINI_API_KEY"] = "dummy"
at.run()
assert not at.exception, at.exception
print("1) 初始載入 OK; buttons:", [b.label for b in at.button])

# 點 Case A 按鈕 -> 文字框應被填入
at.button(key="btn_✅ Case A｜知識庫有明確答案").click().run()
assert not at.exception, at.exception
print("2) Case A 載入後文字框:", repr(at.text_area[0].value))
assert at.text_area[0].value == "收到的商品外包裝有破損，要怎麼處理？"

# 按分析（mock API）
with patch("google.genai.Client", return_value=make_client(payload(fake))) as m:
    click_analyze(at)
    assert not at.exception, at.exception
    called = m.return_value.models.generate_content.call_args
    print("3) API 被呼叫, model =", called.kwargs["model"])
    assert "收到的商品外包裝有破損" in called.kwargs["contents"]
    assert "FAQ013" in called.kwargs["config"].system_instruction
    html = " ".join(x.value for x in at.markdown)
    assert "FAQ013" in html and "測試回覆" in html
    print("   輸出含引用來源與建議回覆 OK")

# Case B
at.button(key="btn_⚠️ Case B｜知識庫完全沒有答案").click().run()
assert at.text_area[0].value == "你們明年會不會推出訂閱制方案？"
with patch("google.genai.Client", return_value=make_client(payload(fake_nokb))):
    click_analyze(at)
    assert not at.exception, at.exception
    html = " ".join(x.value for x in at.markdown)
    assert "無對應來源" in html and "建議轉人工確認" in html
    print("4) Case B(source=null) 渲染 OK")

# 空輸入
at.text_area[0].set_value("").run()
click_analyze(at)
assert any("請輸入問題" in w.value for w in at.warning)
print("5) 空輸入警告 OK")

# JSON 壞掉
at.text_area[0].set_value("測試").run()
with patch("google.genai.Client", return_value=make_client("not json")):
    click_analyze(at)
    assert any("格式不符合預期" in e.value for e in at.error)
    print("6) JSON 格式錯誤處理 OK")

# 引用不存在的 FAQ 編號 -> 應出現警告
with patch("google.genai.Client", return_value=make_client(payload(fake_bad_source))):
    click_analyze(at)
    assert any("FAQ999" in w.value for w in at.warning)
    print("7) 編造來源警告 OK")

# AI 回傳含 HTML 的文字 -> 應被 escape，不會變成真的 HTML
with patch("google.genai.Client", return_value=make_client(payload(dict(fake, suggested_reply="<b>x</b>")))):
    click_analyze(at)
    html = " ".join(x.value for x in at.markdown)
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    print("8) HTML escape OK")
print("ALL TESTS PASSED")
