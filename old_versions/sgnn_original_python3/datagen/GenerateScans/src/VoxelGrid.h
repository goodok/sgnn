#pragma once

#include "mLibInclude.h"


struct Voxel {
	Voxel() {
		sdf = -std::numeric_limits<float>::infinity();
		freeCtr = 0;
		color = vec3uc(0, 0, 0);
		weight = 0;
	}

	float			sdf;
	unsigned int	freeCtr;
	vec3uc			color;
	uchar			weight;
};

class VoxelGrid : public Grid3 < Voxel >
{
public:

	VoxelGrid(const vec3l& dim, const mat4f& worldToGrid, float voxelSize, const OBB3f& gridSceneBounds, float depthMin, float depthMax) : Grid3(dim.x, dim.y, dim.z) {
		m_voxelSize = voxelSize;
		m_depthMin = depthMin;
		m_depthMax = depthMax;
		m_worldToGrid = worldToGrid;
		m_gridToWorld = m_worldToGrid.getInverse();
		m_sceneBoundsGrid = gridSceneBounds;

		m_trunaction = m_voxelSize * 3.0f;
		m_truncationScale = m_voxelSize;
		m_weightUpdate = 1;
	}

	~VoxelGrid() {

	}

	void reset() {
#pragma omp parallel for
		for (int i = 0; i < (int)getNumElements(); i++) {
			getData()[i] = Voxel();
		}
	}

	void integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage);

	//! normalizes the SDFs (divides by the voxel size)
	void normalizeSDFs(float factor = -1.0f) {
		if (factor < 0) factor = 1.0f / m_voxelSize;
		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					Voxel& v = (*this)(i, j, k);
					if (v.sdf != -std::numeric_limits<float>::infinity() && v.sdf != 0.0f) {
						v.sdf *= factor;
					}

				}
			}
		}
		m_voxelSize *= factor;
	}

	//! returns all the voxels on the isosurface
	std::vector<Voxel> getSurfaceVoxels(unsigned int weightThresh, float sdfThresh) const {

		std::vector<Voxel> res;
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					const Voxel& v = (*this)(i, j, k);
					if (v.weight >= weightThresh && std::abs(v.sdf) < sdfThresh) {
						res.push_back(v);
					}
				}
			}
		}

		return res;
	}

	BinaryGrid3 toBinaryGridFree(unsigned int freeThresh) const {
		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					if ((*this)(i, j, k).freeCtr >= freeThresh) {
						res.setVoxel(i, j, k);
					}
				}
			}
		}
		return res;
	}

	BinaryGrid3 toBinaryGridOccupied(unsigned int weightThresh, float sdfThresh) const {

		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					if ((*this)(i, j, k).weight >= weightThresh && std::abs((*this)(i, j, k).sdf) < sdfThresh) {
						res.setVoxel(i, j, k);
					}
				}
			}
		}
		return res;
	}


	void saveToFile(const std::string& filename, bool bSaveSparse, float truncationFactor = -1.0f) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)&m_voxelSize, sizeof(float));
		ofs.write((const char*)m_worldToGrid.getData(), sizeof(mat4f));
		if (bSaveSparse) {
			if (truncationFactor <= 0) truncationFactor = m_trunaction / m_voxelSize;
			std::vector<vec3ui> locations;
			std::vector<float> sdfvalues;
			for (unsigned int z = 0; z < dimZ; z++) {
				for (unsigned int y = 0; y < dimY; y++) {
					for (unsigned int x = 0; x < dimX; x++) {
						const Voxel& v = (*this)(x, y, z);
						if (std::fabs(v.sdf) <= truncationFactor*m_voxelSize) {
							locations.push_back(vec3ui(x, y, z));
							sdfvalues.push_back(v.sdf);
						}
					} // x
				} // y
			} // z
			UINT64 num = (UINT64)locations.size();
			ofs.write((const char*)&num, sizeof(UINT64));
			ofs.write((const char*)locations.data(), sizeof(vec3ui)*locations.size());
			ofs.write((const char*)sdfvalues.data(), sizeof(float)*sdfvalues.size());
		}
		else {
			//dense data
			std::vector<float> values(getNumElements());
			for (unsigned int i = 0; i < getNumElements(); i++) {
				const Voxel& v = getData()[i];
				values[i] = v.sdf;
			}
			ofs.write((const char*)values.data(), sizeof(float)*values.size());
		}
		ofs.close();
	}
	void loadFromFile(const std::string& filename, bool bLoadSparse) {
		std::ifstream ifs(filename, std::ios::binary);
		//metadata
		UINT64 dimX, dimY, dimZ;
		ifs.read((char*)&dimX, sizeof(UINT64));
		ifs.read((char*)&dimY, sizeof(UINT64));
		ifs.read((char*)&dimZ, sizeof(UINT64));
		ifs.read((char*)&m_voxelSize, sizeof(float));
		ifs.read((char*)m_worldToGrid.getData(), sizeof(mat4f));
		m_gridToWorld = m_worldToGrid.getInverse();
		//dense data
		allocate(dimX, dimY, dimZ);
		if (bLoadSparse) {
			UINT64 num;
			ifs.read((char*)&num, sizeof(UINT64));
			//std::cout << filename << ": " << num << std::endl;
			std::vector<vec3ui> locations(num);
			std::vector<float> sdfvalues(num);
			ifs.read((char*)locations.data(), sizeof(vec3ui)*locations.size());
			ifs.read((char*)sdfvalues.data(), sizeof(float)*sdfvalues.size());
			for (unsigned int i = 0; i < locations.size(); i++) {
				Voxel& v = (*this)(locations[i]);
				v.sdf = sdfvalues[i];
				if (v.sdf > -m_voxelSize) v.weight = 1; //just for vis purposes
			}
		}
		else {
			std::vector<float> values(getNumElements());
			ifs.read((char*)values.data(), sizeof(float)*values.size());
			std::vector<vec3uc> colors;
			for (unsigned int i = 0; i < getNumElements(); i++) {
				Voxel& v = getData()[i];
				v.sdf = values[i];
				if (v.sdf >= -m_voxelSize) v.weight = 1;
			}
		}
		ifs.close();
	}

	void saveKnownToFile(const std::string& filename) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)&m_voxelSize, sizeof(float));
		ofs.write((const char*)m_worldToGrid.getData(), sizeof(mat4f));
		std::vector<unsigned char> known(getNumElements());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			const Voxel& v = getData()[i];
			if (v.sdf < -m_voxelSize) known[i] = (unsigned char)std::max(2, std::min(255, (int)(-v.sdf / m_voxelSize) + 1));
			//if (v.sdf < -m_voxelSize)  known[i] = 2; // unknown
			else if (v.sdf <= m_voxelSize)  known[i] = 1; // known occ
			else  known[i] = 0; // known empty
		}
		ofs.write((const char*)known.data(), sizeof(unsigned char)*known.size());
		ofs.close();
	}


	mat4f getGridToWorld() const {
		return m_gridToWorld;
	}

	mat4f getWorldToGrid() const {
		return m_worldToGrid;
	}

	void setVoxelSize(float v) {
		m_voxelSize = v;
	}
	void setWorldToGrid(const mat4f& worldToGrid) {
		m_worldToGrid = worldToGrid;
		m_gridToWorld = worldToGrid.getInverse();
	}

	vec3i worldToVoxel(const vec3f& p) const {
		return math::round((m_worldToGrid * p));
	}

	vec3f worldToVoxelFloat(const vec3f& p) const {
		return (m_worldToGrid * p);
	}

	vec3f voxelToWorld(vec3i& v) const {
		return m_gridToWorld * (vec3f(v));
	}

	float getVoxelSize() const {
		return m_voxelSize;
	}


	bool trilinearInterpolationSimpleFastFast(const vec3f& pos, float& dist, vec3uc& color) const {
		const float oSet = m_voxelSize;
		const vec3f posDual = pos - vec3f(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
		vec3f weight = frac(worldToVoxelFloat(pos));

		dist = 0.0f;
		vec3f colorFloat = vec3f(0.0f, 0.0f, 0.0f);

		Voxel v; vec3f vColor;
		v = getVoxel(posDual + vec3f(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, oSet, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;

		color = vec3uc(math::round(colorFloat.x), math::round(colorFloat.y), math::round(colorFloat.z));

		return true;
	}

	vec3f getSurfaceNormal(size_t x, size_t y, size_t z) const {
		float SDFx = (*this)(x + 1, y, z).sdf - (*this)(x - 1, y, z).sdf;
		float SDFy = (*this)(x, y + 1, z).sdf - (*this)(x, y - 1, z).sdf;
		float SDFz = (*this)(x, y, z + 1).sdf - (*this)(x, y, z - 1).sdf;
		if (SDFx == 0 && SDFy == 0 && SDFz == 0) {// Don't divide by zero!
			return vec3f(SDFx, SDFy, SDFz);
		}
		else {
			return vec3f(SDFx, SDFy, SDFz).getNormalized();
		}
	}

	mat3f getNormalCovariance(int x, int y, int z, int radius, float weightThreshold, float sdfThreshold) const {
		// Compute neighboring surface normals
		std::vector<vec3f> normals;
		for (int k = -radius; k <= radius; k++)
			for (int j = -radius; j <= radius; j++)
				for (int i = -radius; i <= radius; i++)
					if ((*this)(x + i, y + j, z + k).weight >= weightThreshold && std::abs((*this)(x + i, y + j, z + k).sdf) < sdfThreshold)
						normals.push_back(getSurfaceNormal(x + i, y + j, z + k));

		// Find covariance matrix
		float Ixx = 0; float Ixy = 0; float Ixz = 0;
		float Iyy = 0; float Iyz = 0; float Izz = 0;
		for (int i = 0; i < normals.size(); i++) {
			Ixx = Ixx + normals[i].x*normals[i].x;
			Ixy = Ixy + normals[i].x*normals[i].y;
			Ixz = Ixz + normals[i].x*normals[i].z;
			Iyy = Iyy + normals[i].y*normals[i].y;
			Iyz = Iyz + normals[i].y*normals[i].z;
			Izz = Izz + normals[i].z*normals[i].z;
		}
		float scale = 10.0f / ((float)normals.size()); // Normalize and upscale
		return mat3f(Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz)*scale;
	}

	Voxel getVoxel(const vec3f& worldPos) const {
		vec3i voxelPos = worldToVoxel(worldPos);

		if (isValidCoordinate(voxelPos.x, voxelPos.y, voxelPos.z)) {
			return (*this)(voxelPos.x, voxelPos.y, voxelPos.z);
		}
		else {
			return Voxel();
		}
	}


	float getDepthMin() const {
		return m_depthMin;
	}

	float getDepthMax() const {
		return m_depthMax;
	}

	float getTruncation(float d) const {
		return m_trunaction + d * m_truncationScale;
	}

	float getMaxTruncation() const {
		return getTruncation(m_depthMax);
	}
private:

	float frac(float val) const {
		return (val - floorf(val));
	}

	vec3f frac(const vec3f& val) const {
		return vec3f(frac(val.x), frac(val.y), frac(val.z));
	}

	BoundingBox3<int> computeFrustumBounds(const mat4f& intrinsic, const mat4f& rigidTransform, unsigned int width, unsigned int height) const {

		std::vector<vec3f> cornerPoints(8);

		cornerPoints[0] = depthToSkeleton(intrinsic, 0, 0, m_depthMin);
		cornerPoints[1] = depthToSkeleton(intrinsic, width - 1, 0, m_depthMin);
		cornerPoints[2] = depthToSkeleton(intrinsic, width - 1, height - 1, m_depthMin);
		cornerPoints[3] = depthToSkeleton(intrinsic, 0, height - 1, m_depthMin);

		cornerPoints[4] = depthToSkeleton(intrinsic, 0, 0, m_depthMax);
		cornerPoints[5] = depthToSkeleton(intrinsic, width - 1, 0, m_depthMax);
		cornerPoints[6] = depthToSkeleton(intrinsic, width - 1, height - 1, m_depthMax);
		cornerPoints[7] = depthToSkeleton(intrinsic, 0, height - 1, m_depthMax);

		BoundingBox3<int> box;
		for (unsigned int i = 0; i < 8; i++) {

			vec3f pl = math::floor(rigidTransform * cornerPoints[i]);
			vec3f pu = math::ceil(rigidTransform * cornerPoints[i]);
			box.include(worldToVoxel(pl));
			box.include(worldToVoxel(pu));
		}

		box.setMin(math::max(box.getMin(), 0));
		box.setMax(math::min(box.getMax(), vec3i((int)getDimX() - 1, (int)getDimY() - 1, (int)getDimZ() - 1)));

		return box;
	}

	static vec3f depthToSkeleton(const mat4f& intrinsic, unsigned int ux, unsigned int uy, float depth) {
		if (depth == 0.0f || depth == -std::numeric_limits<float>::infinity()) return vec3f(-std::numeric_limits<float>::infinity());

		float x = ((float)ux - intrinsic(0, 2)) / intrinsic(0, 0);
		float y = ((float)uy - intrinsic(1, 2)) / intrinsic(1, 1);

		return vec3f(depth*x, depth*y, depth);
	}

	static vec3f skeletonToDepth(const mat4f& intrinsics, const vec3f& p) {

		float x = (p.x * intrinsics(0, 0)) / p.z + intrinsics(0, 2);
		float y = (p.y * intrinsics(1, 1)) / p.z + intrinsics(1, 2);

		return vec3f(x, y, p.z);
	}

	float m_voxelSize;
	mat4f m_worldToGrid;
	mat4f m_gridToWorld; //inverse of worldToGrid
	float m_depthMin;
	float m_depthMax;
	OBB3f m_sceneBoundsGrid;


	float			m_trunaction;
	float			m_truncationScale;
	unsigned int	m_weightUpdate;
};