import os
from huggingface_hub import HfApi


def deploy():
    print("Initiating Hugging Face SDK deployment...")
    api = HfApi()

    repo_id = "TRISHIT-726/openenv"

    try:
        # 1. Ensure the Space repository exists
        print(f"Creating/Verifying Space: {repo_id} ...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )

        # 2. Upload all assets
        print("Uploading deployment artifacts...")
        current_dir = os.path.dirname(os.path.abspath(__file__))

        api.upload_folder(
            folder_path=current_dir,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Clean OpenEnv deploy — no UI, headless safe",
            ignore_patterns=[
                # Version control
                ".git",
                ".git/*",
                # Python cache
                "__pycache__",
                "**/__pycache__",
                "*.pyc",
                ".hypothesis",
                # Test suite
                "tests/",
                "tests/*",
                # AI scaffolding (not needed on Space)
                ".claude/",
                ".claude/*",
                # Large lock file (Docker re-installs from requirements.txt)
                "uv.lock",
                # Dev-only scripts
                "hf_deploy.py",
                "view_live.py",
                "generate_screenshot.py",
                "validate-submission.sh",
                "inference_v2.py",
                # Debug artifacts
                "tmp_path.txt",
                ".pytest_cache",
            ],
        )

        print("\nSUCCESS: Space created and deployment complete!")
        print(f"Live URL:  https://huggingface.co/spaces/{repo_id}")
        print(f"API docs:  https://{repo_id.replace('/', '-').lower()}.hf.space/docs")

    except Exception as e:
        print(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    deploy()
