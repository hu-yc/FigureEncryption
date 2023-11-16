from api.api_io import app_server
from utils.utils_log import LogFactory

log = LogFactory.get_log('audit')

if __name__ == "__main__":
    log.info("[main] start to run the app server")
    app_server.run(host='0.0.0.0', port=8111, debug=True)
