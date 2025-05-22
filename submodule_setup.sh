# git clone https://<github token>@github.com/TimS-ml/<repo>
# cd <repo>

git submodule init
sed -i 's|git@github.com:|https://<github token>@github.com/|g' .gitmodules
git submodule update
