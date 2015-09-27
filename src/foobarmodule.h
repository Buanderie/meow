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
				float number = 1.5;
       	for( size_t k = 0; k < 1000000; ++k )
      	{
          number*=number;
    		  number = pow(number, number);
	      }
				std::cout << msg.getTimeAsString() << " - " << msg.getContent() << std::endl;
			}
			else
			{
				std::cout << "nope.." << std::endl;
			}
		}

	private:

	protected:


};
