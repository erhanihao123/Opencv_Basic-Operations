import requests
import sys
import time


def check_streamlit(url="http://localhost:8501/_stcore/health"):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True, "Streamlit service is healthy"
        else:
            return False, f"Streamlit service returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Streamlit service check failed: {e}"


def check_nginx(url="http://localhost/health"):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.text.strip() == "OK":
            return True, "Nginx service is healthy"
        else:
            return False, f"Nginx service returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Nginx service check failed: {e}"


def main():
    print("=" * 50)
    print("  Image Annotation Tool - Health Check")
    print("=" * 50)
    print()

    all_healthy = True

    print("Checking Streamlit service...")
    healthy, message = check_streamlit()
    print(f"  {'✓' if healthy else '✗'} {message}")
    if not healthy:
        all_healthy = False

    print()
    print("Checking Nginx service...")
    healthy, message = check_nginx()
    print(f"  {'✓' if healthy else '✗'} {message}")
    if not healthy:
        all_healthy = False

    print()
    print("=" * 50)
    if all_healthy:
        print("  All services are healthy!")
        print("=" * 50)
        sys.exit(0)
    else:
        print("  Some services are not healthy!")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()