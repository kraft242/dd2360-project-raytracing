#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"
#include "fp_data_type.h"

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, FpDataType tmin, FpDataType tmax, hit_record& rec) const;
        hitable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const ray& r, FpDataType t_min, FpDataType t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        FpDataType closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
