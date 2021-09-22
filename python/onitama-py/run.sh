#!/bin/bash
source "/usr/local/bin/virtualenvwrapper.sh" && workon mlp && cd onitama && flask run

# if you have no venv and just system installed python libraries, try this
# cd backend && backend run
