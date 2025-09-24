module.exports = {
  apps: [
    {
      name: 'frontend-service',
      script: 'python3',
      args: '-m uvicorn app:app --host 0.0.0.0 --port 8080',
      cwd: '/home/user/webapp',
      env: {
        PORT: '8080',
        PYTHONPATH: '/home/user/webapp'
      },
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      log_file: './logs/frontend.log',
      error_file: './logs/frontend-error.log',
      out_file: './logs/frontend-out.log'
    },
    {
      name: 'main-api-service',
      script: 'python3',
      args: '-m uvicorn app:app --host 0.0.0.0 --port 8000',
      cwd: '/home/user/webapp',
      env: {
        PORT: '8000',
        PYTHONPATH: '/home/user/webapp'
      },
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      log_file: './logs/api.log',
      error_file: './logs/api-error.log',
      out_file: './logs/api-out.log'
    }
  ]
};