"""Entry point for Vercel (it looks for a Flask `app` in app.py). Locally you can also run: python app.py"""
from web_app import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"AppendiScope is running - open http://localhost:{port}  (press Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
