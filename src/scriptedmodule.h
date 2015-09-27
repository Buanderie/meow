
//
#include "module.h"
#include "scriptinterpreter.h"

class ScriptedModule : public Module
{
	friend class ScriptInterpreter;
	public:
		ScriptedModule( const std::string& sourceCode="" )
			:Module(), _sourceCode( sourceCode )
		{
			_interpreter = new ScriptInterpreter(this);
		}
		
		virtual ~ScriptedModule()
		{
			delete _interpreter;
		}
		
		virtual void tick()
		{
			_interpreter->callTick();
		}
		
		void test()
		{
			std::cout << "test?" << std::endl;
		}
		
		void setScript( const std::string& sourceCode )
		{
			_sourceCode = sourceCode;
		}
		
	private:
		std::string 				_sourceCode;
		ScriptInterpreter* 	_interpreter;
		
		virtual void emit( Message& msg )
		{
			Module::emit(msg);
		}

                Message scriptReceive( bool isBlocking=true )
                {
                        Message msg;
                        // std::cout << "before: " << msg.getContent() << std::endl;
                        bool o = Module::receive( msg, isBlocking );
                        // std::cout << "after: " << msg.getContent() << std::endl;
                        // std::cout << "ScriptedModule.receive : " << msg.getContent() << std::endl;
                        return msg;
                }
	protected:

};

