ps -ef | grep ljj | grep python3 | grep -v grep | awk '{print $2}' | xargs kill -9

