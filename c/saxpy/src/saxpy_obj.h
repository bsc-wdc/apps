#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>


using namespace std;
using namespace boost;
using namespace serialization;


class saxpy_obj {

public:
    float *value;
    int N;

    saxpy_obj(){};

    void init(int val);

private:
    friend class::serialization::access;
    template<class Archive>

    void serialize(Archive & ar, const unsigned int version) {
        ar & N;
        if (Archive::is_loading::value){
            value = new float[N];
        }
        fflush(NULL);
        ar & make_array<float>(value, N);
	fflush(NULL);
    }
};
