# **************************************************************************** #
#                                                                              #
# Copyright (C) 2020 Urgent Computing Integrated Services 4 EQ (UCIS4EQ)       #
# This file is part of the CHEESE project.                                     #
#                                                                              #
# Barcelona Supercomputing Center - Centro Nacional de Supercomputacion        #
#   Nexus II Building                                                          #
#   c/ Jordi Girona, 29                                                        #
#   08034 Barcelona, Spain                                                     #
#   Phone: (+34) 93 413 77 16 (Switchboard)                                    #
#   Fax:   (+34) 93 413 77 21                                                  #
#                                                                              #
# Juan Esteban Rodriguez                                                       #
# January 2020                                                                 #
# Computer Applications in Science & Engineering                               #
# BSC-CNS                                                                      #
#                                                                              #
# **************************************************************************** #

# ###############################################################################
# UCIS4EQ module publication 
# ###############################################################################

# Check the latest UCIS4EQ version published
pip install yolk3k

if [ "$(yolk -V ucis4eq | awk '{print $2}')" = "$CI_COMMIT_TAG" ]
then
  echo "UCIS4EQ is already up to date"
  exit 0
fi

# Setup the environment 

# Configure code

# Publish UCIS4EQ on Pypi
pip install twine #==3.0.0
pip install --upgrade keyrings.alt
cp -R code/components deployment/pypi/ucis4eq
find deployment/pypi/ucis4eq/ -type d -name "__pycache__" -print0 | xargs -0 rm -fr
cd deployment/pypi/ || printf "%s\n" "Directory unknowni: deployment/pypi/"
sed -i 's/VERSION/'"$CI_COMMIT_TAG"'/'  setup.py
python setup.py sdist
twine upload dist/*
sed -i 's/'"$CI_COMMIT_TAG"'/VERSION/'  setup.py
