git co --orphan public
git rm --cached -r c java python
cat public_apps.txt | xargs git add
git ci -m 'added public apps'
