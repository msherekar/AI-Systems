#!/usr/bin/env python3
"""
Script for deploying the sentiment analysis model API.
"""
import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_docker_installed():
    """
    Check if Docker is installed.
    
    Returns:
        bool: True if Docker is installed, False otherwise
    """
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Docker is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed")
        return False

def check_docker_compose_installed():
    """
    Check if Docker Compose is installed.
    
    Returns:
        bool: True if Docker Compose is installed, False otherwise
    """
    try:
        subprocess.run(["docker-compose", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Docker Compose is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker Compose is not installed")
        return False

def deploy_with_docker_compose():
    """
    Deploy the application using Docker Compose.
    """
    try:
        logger.info("Building and starting containers with Docker Compose")
        subprocess.run(["docker-compose", "up", "-d", "--build"], check=True, cwd=project_root)
        logger.info("Containers started successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error deploying with Docker Compose: {e}")
        return False

def deploy_with_docker(port=8786):
    """
    Deploy the application using Docker.
    
    Args:
        port (int): Port to expose
    """
    try:
        logger.info("Building Docker image")
        subprocess.run(["docker", "build", "-t", "sentiment-analysis:latest", "."], check=True, cwd=project_root)
        
        logger.info(f"Starting Docker container on port {port}")
        subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:8786", "--name", "sentiment-analysis-api", "sentiment-analysis:latest"],
            check=True,
            cwd=project_root
        )
        
        logger.info("Container started successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error deploying with Docker: {e}")
        return False

def deploy_locally(debug=False):
    """
    Deploy the application locally without Docker.
    
    Args:
        debug (bool): Whether to run in debug mode
    """
    try:
        logger.info("Starting local deployment")
        
        # Set environment variables
        env = os.environ.copy()
        env["FLASK_DEBUG"] = "True" if debug else "False"
        
        # Start the Flask application
        logger.info("Starting Flask application")
        deployment_script = os.path.join(project_root, "src", "deployment", "deployment.py")
        
        subprocess.Popen([sys.executable, deployment_script], env=env)
        logger.info("Flask application started successfully")
        return True
    except Exception as e:
        logger.error(f"Error deploying locally: {e}")
        return False

def main(args):
    """
    Main function for deploying the application.
    
    Args:
        args: Command-line arguments
    """
    if args.deployment_method == "docker-compose":
        if not check_docker_installed() or not check_docker_compose_installed():
            logger.error("Docker or Docker Compose not installed. Please install them to use this deployment method.")
            return
        
        success = deploy_with_docker_compose()
    elif args.deployment_method == "docker":
        if not check_docker_installed():
            logger.error("Docker not installed. Please install Docker to use this deployment method.")
            return
        
        success = deploy_with_docker(port=args.port)
    else:  # local
        success = deploy_locally(debug=args.debug)
    
    if success:
        logger.info("Deployment completed successfully")
        if args.deployment_method in ["docker", "docker-compose"]:
            logger.info(f"API available at: http://localhost:{args.port}")
        else:
            logger.info(f"API available at: http://localhost:8786")
    else:
        logger.error("Deployment failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the sentiment analysis model API")
    parser.add_argument(
        "--deployment-method",
        type=str,
        choices=["local", "docker", "docker-compose"],
        default="local",
        help="Deployment method"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8786,
        help="Port to expose the API on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (local deployment only)"
    )
    
    args = parser.parse_args()
    main(args) 