#pragma once

#include "module.h"

class FoobarModule : public Module
{
	public:
		FoobarModule()
		{

		}

		virtual ~FoobarModule()
		{

		}

		void tick()
		{
			Message msg;
			if( receive( msg ) )
			{
				std::cout << msg.getTimeAsString() << " - " << msg.getContent() << std::endl;
			}
		}

	private:

	protected:


};
