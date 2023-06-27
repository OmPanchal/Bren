echo off

echo Available Branches:
git branch

set /p branch=What branch would you like to commit to?: 

git checkout %branch%

git rm -rf --cached .

git add .

git commit -m "bren"
git push -u origin %branch%