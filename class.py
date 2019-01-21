class Person:
    def setName(self,name):
        self.name = name
    def getNames(self):
        return self.name
    def greet(self):
        print("Hello, I am %s." % self.name)

person1 = Person()
person2 = Person()
person3 = Person()
person4 = Person()
person5 = Person()

person1.setName('Se Yeol Rhyou')
person2.setName('Yoon Sung Hyeon')

person1.greet()
person2.greet()