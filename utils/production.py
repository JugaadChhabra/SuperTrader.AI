"""
Production Deployment Configuration - Phase 5

Production-ready deployment configuration for SuperTrader.AI with
comprehensive settings, security considerations, and monitoring setup.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import yaml
import json

from utils.config import get_config
from utils.performance import get_performance_optimizer


@dataclass
class SecurityConfig:
    """Security configuration for production deployment"""
    api_key_required: bool = True
    max_requests_per_minute: int = 100
    allowed_ips: List[str] = None
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    enable_request_logging: bool = True
    
    def __post_init__(self):
        if self.allowed_ips is None:
            self.allowed_ips = ["127.0.0.1", "localhost"]


@dataclass
class DatabaseConfig:
    """Database configuration for production"""
    host: str = "localhost"
    port: int = 5432
    database: str = "supertrader"
    username: str = "supertrader_user"
    password: str = ""  # Should be set via environment variable
    connection_pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    enable_ssl: bool = True


@dataclass
class RedisConfig:
    """Redis configuration for caching and real-time data"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""  # Should be set via environment variable
    database: int = 0
    connection_pool_size: int = 50
    socket_timeout: int = 5
    enable_clustering: bool = False


@dataclass
class LoggingConfig:
    """Comprehensive logging configuration"""
    level: str = "INFO"
    log_dir: str = "/var/log/supertrader"
    max_file_size_mb: int = 100
    backup_count: int = 10
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    enable_structured_logging: bool = True
    enable_audit_logging: bool = True
    enable_performance_logging: bool = True
    syslog_enabled: bool = False
    syslog_address: str = "/dev/log"


@dataclass
class AlertingConfig:
    """Alerting and notification configuration"""
    enable_email_alerts: bool = True
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""  # Environment variable
    alert_recipients: List[str] = None
    
    enable_slack_alerts: bool = False
    slack_webhook_url: str = ""
    
    enable_sms_alerts: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    alert_phone_numbers: List[str] = None
    
    critical_alert_cooldown_minutes: int = 15
    warning_alert_cooldown_minutes: int = 60
    
    def __post_init__(self):
        if self.alert_recipients is None:
            self.alert_recipients = []
        if self.alert_phone_numbers is None:
            self.alert_phone_numbers = []


@dataclass
class DeploymentConfig:
    """Main deployment configuration"""
    environment: str = "production"  # development, staging, production
    debug: bool = False
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    worker_timeout: int = 300
    max_requests_per_worker: int = 1000
    
    # Performance settings
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Feature flags
    enable_phase4_features: bool = True
    enable_real_time_monitoring: bool = True
    enable_auto_scaling: bool = False
    
    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    
    # Health checks
    health_check_interval_seconds: int = 30
    startup_timeout_seconds: int = 120


class ProductionDeploymentManager:
    """
    Manager for production deployment configuration and setup
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.base_config = get_config()
        
        # Load environment-specific configurations
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.logging = LoggingConfig()
        self.alerting = AlertingConfig()
        self.deployment = DeploymentConfig(environment=environment)
        
        # Override with environment variables
        self._load_environment_variables()
        
        # Setup logging first
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Production deployment manager initialized for {environment}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # Database configuration
        self.database.password = os.getenv('DB_PASSWORD', '')
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.port = int(os.getenv('DB_PORT', self.database.port))
        
        # Redis configuration
        self.redis.password = os.getenv('REDIS_PASSWORD', '')
        self.redis.host = os.getenv('REDIS_HOST', self.redis.host)
        self.redis.port = int(os.getenv('REDIS_PORT', self.redis.port))
        
        # Security configuration
        api_key = os.getenv('API_KEY')
        if api_key:
            os.environ['SUPERTRADER_API_KEY'] = api_key
        
        # Alerting configuration
        self.alerting.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.alerting.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.alerting.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        
        # Deployment configuration
        if os.getenv('DEBUG', '').lower() in ['true', '1', 'yes']:
            self.deployment.debug = True
        
        self.deployment.port = int(os.getenv('PORT', self.deployment.port))
        self.deployment.workers = int(os.getenv('WORKERS', self.deployment.workers))
    
    def _setup_logging(self):
        """Setup comprehensive logging for production"""
        # Create log directory
        log_dir = Path(self.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.log_format
        )
        
        # Add file handlers
        from logging.handlers import RotatingFileHandler, SysLogHandler
        
        # Main application log
        main_handler = RotatingFileHandler(
            log_dir / "supertrader.log",
            maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
            backupCount=self.logging.backup_count
        )
        main_handler.setFormatter(logging.Formatter(self.logging.log_format))
        
        # Error log
        error_handler = RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
            backupCount=self.logging.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(self.logging.log_format))
        
        # Performance log
        if self.logging.enable_performance_logging:
            perf_handler = RotatingFileHandler(
                log_dir / "performance.log",
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            perf_handler.setFormatter(logging.Formatter(self.logging.log_format))
            
            # Add performance logger
            perf_logger = logging.getLogger('performance')
            perf_logger.addHandler(perf_handler)
        
        # Syslog handler
        if self.logging.syslog_enabled:
            try:
                syslog_handler = SysLogHandler(address=self.logging.syslog_address)
                syslog_handler.setFormatter(logging.Formatter('supertrader: %(message)s'))
                logging.getLogger().addHandler(syslog_handler)
            except Exception as e:
                logging.warning(f"Failed to setup syslog handler: {e}")
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(main_handler)
        root_logger.addHandler(error_handler)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate production configuration"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required environment variables
        required_env_vars = ['DB_PASSWORD']
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                validation_results['errors'].append(f"Missing required environment variable: {env_var}")
        
        # Validate database connection
        if not self.database.password:
            validation_results['errors'].append("Database password not configured")
        
        # Validate security settings
        if self.deployment.environment == 'production':
            if not self.security.api_key_required:
                validation_results['warnings'].append("API key authentication disabled in production")
            
            if self.deployment.debug:
                validation_results['errors'].append("Debug mode enabled in production")
            
            if "0.0.0.0" in self.security.allowed_ips:
                validation_results['warnings'].append("Open IP access configured")
        
        # Validate file permissions and directories
        log_dir = Path(self.logging.log_dir)
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                validation_results['errors'].append(f"Cannot create log directory: {log_dir}")
        
        # Check SSL configuration for production
        if self.deployment.environment == 'production':
            if not self.security.ssl_cert_path:
                validation_results['warnings'].append("SSL certificate not configured for production")
        
        validation_results['valid'] = len(validation_results['errors']) == 0
        return validation_results
    
    def generate_deployment_files(self, output_dir: str = "deployment"):
        """Generate deployment configuration files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate Docker configuration
        self._generate_dockerfile(output_path)
        self._generate_docker_compose(output_path)
        
        # Generate systemd service file
        self._generate_systemd_service(output_path)
        
        # Generate nginx configuration
        self._generate_nginx_config(output_path)
        
        # Generate environment file template
        self._generate_env_template(output_path)
        
        # Generate monitoring configuration
        self._generate_monitoring_config(output_path)
        
        self.logger.info(f"Deployment files generated in {output_path}")
    
    def _generate_dockerfile(self, output_path: Path):
        """Generate Dockerfile for production deployment"""
        dockerfile_content = f"""
# SuperTrader.AI Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash supertrader
RUN chown -R supertrader:supertrader /app
USER supertrader

# Create log directory
RUN mkdir -p {self.logging.log_dir}

# Expose port
EXPOSE {self.deployment.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{self.deployment.port}/health || exit 1

# Run application
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:{self.deployment.port}", "--workers", "{self.deployment.workers}", "--timeout", "{self.deployment.worker_timeout}", "app:create_app()"]
"""
        
        with open(output_path / "Dockerfile", 'w') as f:
            f.write(dockerfile_content.strip())
    
    def _generate_docker_compose(self, output_path: Path):
        """Generate Docker Compose configuration"""
        compose_content = f"""
version: '3.8'

services:
  supertrader:
    build: .
    ports:
      - "{self.deployment.port}:{self.deployment.port}"
    environment:
      - ENVIRONMENT={self.deployment.environment}
      - DB_HOST={self.database.host}
      - DB_PORT={self.database.port}
      - REDIS_HOST={self.redis.host}
      - REDIS_PORT={self.redis.port}
    env_file:
      - .env
    volumes:
      - ./logs:{self.logging.log_dir}
      - ./cache:/app/cache
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.deployment.port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB={self.database.database}
      - POSTGRES_USER={self.database.username}
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "{self.database.port}:{self.database.port}"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "{self.redis.port}:{self.redis.port}"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - supertrader
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
"""
        
        with open(output_path / "docker-compose.yml", 'w') as f:
            f.write(compose_content.strip())
    
    def _generate_systemd_service(self, output_path: Path):
        """Generate systemd service file"""
        service_content = f"""
[Unit]
Description=SuperTrader.AI Trading System
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=supertrader
Group=supertrader
WorkingDirectory=/opt/supertrader
Environment=ENVIRONMENT={self.deployment.environment}
EnvironmentFile=/opt/supertrader/.env
ExecStart=/opt/supertrader/venv/bin/python -m gunicorn --bind 127.0.0.1:{self.deployment.port} --workers {self.deployment.workers} --timeout {self.deployment.worker_timeout} app:create_app()
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=supertrader

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths={self.logging.log_dir} /opt/supertrader/cache

# Resource limits
LimitNOFILE=65536
MemoryMax={self.deployment.max_memory_mb}M
CPUQuota={int(self.deployment.max_cpu_percent)}%

[Install]
WantedBy=multi-user.target
"""
        
        with open(output_path / "supertrader.service", 'w') as f:
            f.write(service_content.strip())
    
    def _generate_nginx_config(self, output_path: Path):
        """Generate Nginx configuration"""
        nginx_content = f"""
upstream supertrader {{
    server 127.0.0.1:{self.deployment.port};
}}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate={self.security.max_requests_per_minute}r/m;

server {{
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name _;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req zone=api burst=20 nodelay;

    # Proxy configuration
    location / {{
        proxy_pass http://supertrader;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}

    # Health check endpoint
    location /health {{
        proxy_pass http://supertrader;
        access_log off;
    }}

    # Static files (if any)
    location /static {{
        alias /opt/supertrader/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}
}}
"""
        
        with open(output_path / "nginx.conf", 'w') as f:
            f.write(nginx_content.strip())
    
    def _generate_env_template(self, output_path: Path):
        """Generate environment variable template"""
        env_content = f"""
# SuperTrader.AI Environment Configuration
ENVIRONMENT={self.deployment.environment}
DEBUG={"true" if self.deployment.debug else "false"}

# Database Configuration
DB_HOST={self.database.host}
DB_PORT={self.database.port}
DB_NAME={self.database.database}
DB_USERNAME={self.database.username}
DB_PASSWORD=your_secure_password_here

# Redis Configuration
REDIS_HOST={self.redis.host}
REDIS_PORT={self.redis.port}
REDIS_PASSWORD=your_redis_password_here

# API Security
API_KEY=your_secure_api_key_here

# SMTP Configuration (for alerts)
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_password

# Slack Integration (optional)
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Twilio Configuration (optional)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token

# Performance Settings
ENABLE_CACHING={"true" if self.deployment.enable_caching else "false"}
CACHE_TTL_SECONDS={self.deployment.cache_ttl_seconds}

# Resource Limits
MAX_MEMORY_MB={self.deployment.max_memory_mb}
MAX_CPU_PERCENT={self.deployment.max_cpu_percent}
"""
        
        with open(output_path / ".env.template", 'w') as f:
            f.write(env_content.strip())
    
    def _generate_monitoring_config(self, output_path: Path):
        """Generate monitoring configuration"""
        monitoring_config = {
            'version': '1.0',
            'environment': self.deployment.environment,
            'monitoring': {
                'health_checks': {
                    'interval_seconds': self.deployment.health_check_interval_seconds,
                    'timeout_seconds': 10,
                    'endpoints': [
                        '/health',
                        '/health/deep',
                        '/metrics'
                    ]
                },
                'metrics': {
                    'enabled': self.deployment.enable_performance_monitoring,
                    'collection_interval_seconds': 60,
                    'retention_days': 30
                },
                'alerting': asdict(self.alerting),
                'logging': asdict(self.logging)
            }
        }
        
        with open(output_path / "monitoring.yml", 'w') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False, indent=2)
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment configuration summary"""
        return {
            'environment': self.deployment.environment,
            'configuration': {
                'deployment': asdict(self.deployment),
                'security': asdict(self.security),
                'database': {**asdict(self.database), 'password': '***'},
                'redis': {**asdict(self.redis), 'password': '***'},
                'logging': asdict(self.logging),
                'alerting': {**asdict(self.alerting), 
                           'smtp_password': '***',
                           'twilio_auth_token': '***'}
            },
            'validation': self.validate_configuration()
        }


def setup_production_deployment(environment: str = "production", output_dir: str = "deployment"):
    """Setup production deployment configuration"""
    manager = ProductionDeploymentManager(environment)
    
    # Validate configuration
    validation = manager.validate_configuration()
    if not validation['valid']:
        print("Configuration validation failed:")
        for error in validation['errors']:
            print(f"  ERROR: {error}")
        for warning in validation['warnings']:
            print(f"  WARNING: {warning}")
        return False
    
    # Generate deployment files
    manager.generate_deployment_files(output_dir)
    
    # Print summary
    summary = manager.get_deployment_summary()
    print(f"Production deployment configured for {environment}")
    print(f"Deployment files generated in: {output_dir}")
    
    return True


if __name__ == "__main__":
    # Setup production deployment
    success = setup_production_deployment()
    
    if success:
        print("Production deployment setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and update .env.template with your actual credentials")
        print("2. Generate SSL certificates for HTTPS")
        print("3. Test deployment in staging environment")
        print("4. Deploy to production using docker-compose or systemd")
    else:
        print("Production deployment setup failed!")