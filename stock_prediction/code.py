import web
render = web.template.render('templates/')
urls=(
	'/','index'
	)

app=web.application(urls,globals())

class index:
	def GET(self):
		return 'hello,world'


if  __name__=='__main__':

	app.run()