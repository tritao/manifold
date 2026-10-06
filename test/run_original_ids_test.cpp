#include <gtest/gtest.h>
#include <manifold/manifold.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

using manifold::Manifold;
using manifold::MeshGL64;

static void compare_runs(const MeshGL64& before, const MeshGL64& after,
                         const std::vector<uint32_t>& ids) {
  ASSERT_TRUE(before.vertProperties == after.vertProperties);
  ASSERT_TRUE(before.mergeFromVert == after.mergeFromVert);
  ASSERT_TRUE(before.mergeToVert == after.mergeToVert);
  ASSERT_TRUE(before.runOriginalID.size() == after.runOriginalID.size());
  ASSERT_EQ(before.faceID.size(), after.faceID.size());
  constexpr size_t TriangleCorners = 3;
  constexpr size_t TransformElements = 12;
  for (size_t old = 0; old < ids.size(); ++old) {
    auto found = std::find(after.runOriginalID.begin(),
                           after.runOriginalID.end(), ids[old]);
    ASSERT_TRUE(found != after.runOriginalID.end());
    const size_t mapped = size_t(found - after.runOriginalID.begin());
    const size_t oldStart = before.runIndex[old],
                 newStart = after.runIndex[mapped];
    const size_t length = before.runIndex[old + 1] - oldStart;
    ASSERT_TRUE(after.runIndex[mapped + 1] - newStart == length);
    ASSERT_TRUE(std::equal(before.triVerts.begin() + oldStart,
                           before.triVerts.begin() + oldStart + length,
                           after.triVerts.begin() + newStart));
    if (!before.faceID.empty())
      ASSERT_TRUE(std::equal(
          before.faceID.begin() + oldStart / TriangleCorners,
          before.faceID.begin() + (oldStart + length) / TriangleCorners,
          after.faceID.begin() + newStart / TriangleCorners));
    ASSERT_TRUE(before.runFlags[old] == after.runFlags[mapped]);
    for (size_t element = 0; element < TransformElements; ++element) {
      constexpr size_t LinearElements = 9;
      constexpr size_t DiagonalStride = 4;
      const double expected =
          before.runTransform.empty()
              ? (element < LinearElements && element % DiagonalStride == 0
                     ? 1.0
                     : 0.0)
              : before.runTransform[old * TransformElements + element];
      ASSERT_TRUE(after.runTransform[mapped * TransformElements + element] ==
                  expected);
    }
  }
}

TEST(Manifold, WithRunOriginalIDsPreservesProvenance) {
  const Manifold leaf = Manifold::Cube({1, 2, 3}).AsOriginal().TriID2FaceID();
  const auto leafMesh = leaf.GetMeshGL64();
  const auto first = Manifold::ReserveIDs(1);
  const auto rekeyedLeaf = leaf.WithRunOriginalIDs({first});
  ASSERT_TRUE(rekeyedLeaf.Status() == Manifold::Error::NoError);
  compare_runs(leafMesh, rekeyedLeaf.GetMeshGL64(), {first});
  ASSERT_TRUE(leaf.GetMeshGL64().runOriginalID == leafMesh.runOriginalID);

  const auto normals = leaf.CalculateNormals().Translate({0, 1, 0});
  const auto normalsMesh = normals.GetMeshGL64();
  const auto normalsID =
      Manifold::ReserveIDs(uint32_t(normalsMesh.runOriginalID.size()));
  ASSERT_TRUE(normalsMesh.runOriginalID.size() == 1);
  compare_runs(normalsMesh,
               normals.WithRunOriginalIDs({normalsID}).GetMeshGL64(),
               {normalsID});

  // Multi-leaf prototypes and two copies with the very same transform.
  const auto multi = Manifold::BatchBoolean(
      {leaf, leaf.Translate({4, 0, 0}), leaf}, manifold::OpType::Add);
  const auto before = multi.GetMeshGL64();
  ASSERT_TRUE(before.runOriginalID.size() == 3);

  const auto block = Manifold::ReserveIDs(3);
  const std::vector<uint32_t> ids = {block + 2, block, block + 1};
  const auto mapped = multi.WithRunOriginalIDs(ids);
  ASSERT_TRUE(mapped.Status() == Manifold::Error::NoError);
  compare_runs(before, mapped.GetMeshGL64(), ids);
  ASSERT_TRUE(multi.GetMeshGL64().runOriginalID == before.runOriginalID);
  const auto shifted = mapped.Translate({0, 5, 0});
  const auto composed =
      Manifold::BatchBoolean({mapped, shifted}, manifold::OpType::Add);
  ASSERT_TRUE(composed.GetMeshGL64().runOriginalID.size() == 6);

  // Explicit zero-face ancestor relation survives import and relabeling.
  auto zeroInput = leafMesh;
  zeroInput.runOriginalID.push_back(Manifold::ReserveIDs(1));
  zeroInput.runIndex.push_back(zeroInput.runIndex.back());
  zeroInput.runFlags.push_back(0);
  const Manifold zeroAncestor(zeroInput);
  ASSERT_TRUE(zeroAncestor.Status() == Manifold::Error::NoError);
  const auto zeroMesh = zeroAncestor.GetMeshGL64();
  ASSERT_TRUE(zeroMesh.runOriginalID.size() == 2);
  ASSERT_TRUE(
      std::adjacent_find(zeroMesh.runIndex.begin(), zeroMesh.runIndex.end()) !=
      zeroMesh.runIndex.end());
  const auto zeroBlock = Manifold::ReserveIDs(2);
  compare_runs(
      zeroMesh,
      zeroAncestor.WithRunOriginalIDs({zeroBlock, zeroBlock + 1}).GetMeshGL64(),
      {zeroBlock, zeroBlock + 1});

  const auto boolean = leaf + leaf.Translate({0.5, 0, 0});
  const auto booleanMesh = boolean.GetMeshGL64();
  const auto booleanBlock =
      Manifold::ReserveIDs(uint32_t(booleanMesh.runOriginalID.size()));
  std::vector<uint32_t> booleanIds;
  for (size_t run = 0; run < booleanMesh.runOriginalID.size(); ++run)
    booleanIds.push_back(booleanBlock + uint32_t(run));
  compare_runs(booleanMesh,
               boolean.WithRunOriginalIDs(booleanIds).GetMeshGL64(),
               booleanIds);
  const auto result = boolean.WithRunOriginalIDs(booleanIds) -
                      Manifold::Cube({0.2, 4, 4}).Translate({0.6, -1, -1});
  ASSERT_TRUE(result.Status() == Manifold::Error::NoError);
  const auto output = result.GetMeshGL64();
  for (uint32_t id : booleanIds)
    ASSERT_TRUE(std::find(output.runOriginalID.begin(),
                          output.runOriginalID.end(),
                          id) != output.runOriginalID.end());

  ASSERT_TRUE(leaf.WithRunOriginalIDs({}).Status() ==
              Manifold::Error::InvalidConstruction);
  ASSERT_TRUE(multi.WithRunOriginalIDs({block, block, block + 1}).Status() ==
              Manifold::Error::InvalidConstruction);
  ASSERT_TRUE(leaf.WithRunOriginalIDs({std::numeric_limits<uint32_t>::max()})
                  .Status() == Manifold::Error::InvalidConstruction);
  ASSERT_TRUE(Manifold().WithRunOriginalIDs({}).IsEmpty());
  ASSERT_TRUE(Manifold().WithRunOriginalIDs({first}).Status() ==
              Manifold::Error::InvalidConstruction);
}

TEST(Manifold, WithRunOriginalIDsWithoutFaceIDs) {
  const auto leaf = Manifold::Cube({1, 2, 3}).AsOriginal();
  const auto before = leaf.GetMeshGL64();
  ASSERT_TRUE(before.faceID.empty());
  const auto id = Manifold::ReserveIDs(1);
  const auto mapped = leaf.WithRunOriginalIDs({id});
  ASSERT_EQ(mapped.Status(), Manifold::Error::NoError);
  compare_runs(before, mapped.GetMeshGL64(), {id});
  const auto invalid = Manifold::Cube({-1, 2, 3});
  ASSERT_NE(invalid.Status(), Manifold::Error::NoError);
  EXPECT_EQ(invalid.WithRunOriginalIDs({id}).Status(), invalid.Status());
}
