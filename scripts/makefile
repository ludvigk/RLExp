deploy:
	git add -A
	-git rm $$(git ls-files --deleted) 2> /dev/null
	git commit --allow-empty --no-verify --no-gpg-sign -m "TEMPORARY DEPLOY COMMIT"
	-git push --force --no-verify ${DEPLOY_TARGET} HEAD:deploy