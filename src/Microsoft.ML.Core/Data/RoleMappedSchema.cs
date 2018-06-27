// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This contains information about a column in an <see cref="IDataView"/>. It is essentially a convenience cache
    /// containing the name, column index, and column type for the column. The intended usage is that users of <see cref="RoleMappedSchema"/>
    /// to get the column index and type associated with 
    /// </summary>
    public sealed class ColumnInfo
    {
        public readonly string Name;
        public readonly int Index;
        public readonly ColumnType Type;

        private ColumnInfo(string name, int index, ColumnType type)
        {
            Name = name;
            Index = index;
            Type = type;
        }

        /// <summary>
        /// Create a ColumnInfo for the column with the given name in the given schema. Throws if the name
        /// doesn't map to a column.
        /// </summary>
        public static ColumnInfo CreateFromName(ISchema schema, string name, string descName)
        {
            ColumnInfo colInfo;
            if (!TryCreateFromName(schema, name, out colInfo))
                throw Contracts.ExceptParam(nameof(name), "{0} column '{1}' not found", descName, name);

            return colInfo;
        }

        /// <summary>
        /// Tries to create a ColumnInfo for the column with the given name in the given schema. Returns
        /// false if the name doesn't map to a column.
        /// </summary>
        public static bool TryCreateFromName(ISchema schema, string name, out ColumnInfo colInfo)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckNonEmpty(name, nameof(name));

            colInfo = null;
            int index;
            if (!schema.TryGetColumnIndex(name, out index))
                return false;

            colInfo = new ColumnInfo(name, index, schema.GetColumnType(index));
            return true;
        }

        /// <summary>
        /// Creates a ColumnInfo for the column with the given column index. Note that the name
        /// of the column might actually map to a different column, so this should be used with care
        /// and rarely.
        /// </summary>
        public static ColumnInfo CreateFromIndex(ISchema schema, int index)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(0 <= index && index < schema.ColumnCount, nameof(index));

            return new ColumnInfo(schema.GetColumnName(index), index, schema.GetColumnType(index));
        }
    }

    /// <summary>
    /// Encapsulates an <see cref="ISchema"/> plus column role mapping information. The purpose of role mappings is to
    /// provide information on what the intended usage is for. That is: while a given data view may have a column named
    /// "Features", by itself that is insufficient: the trainer must be fed a role mapping that says that the role
    /// mapping for features is filled by that "Features" column. This allows things like columns not named "Features"
    /// to actually fill that role (as opposed to insisting on a hard coding, or having every trainer have to be
    /// individually configured). Also, by being a one-to-many mapping, it is a way for learners that can consume
    /// multiple features columns to consume that information.
    ///
    /// This class has convenience fields for several common column roles (se.g., <see cref="Feature"/>, <see
    /// cref="Label"/>), but can hold an arbitrary set of column infos. The convenience fields are non-null iff there is
    /// a unique column with the corresponding role. When there are no such columns or more than one such column, the
    /// field is null. The <see cref="Has"/>, <see cref="HasUnique"/>, and <see cref="HasMultiple"/> methods provide
    /// some cardinality information. Note that all columns assigned roles are guaranteed to be non-hidden in this
    /// schema.
    /// </summary>
    public sealed class RoleMappedSchema
    {
        private const string FeatureString = "Feature";
        private const string LabelString = "Label";
        private const string GroupString = "Group";
        private const string WeightString = "Weight";
        private const string NameString = "Name";
        private const string FeatureContributionsString = "FeatureContributions";

        public struct ColumnRole
        {
            public static ColumnRole Feature => FeatureString;
            public static ColumnRole Label => LabelString;
            public static ColumnRole Group => GroupString;
            public static ColumnRole Weight => WeightString;
            public static ColumnRole Name => NameString;
            public static ColumnRole FeatureContributions => FeatureContributionsString;

            public readonly string Value;

            public ColumnRole(string value)
            {
                Contracts.CheckNonEmpty(value, nameof(value));
                Value = value;
            }

            public static implicit operator ColumnRole(string value)
            {
                return new ColumnRole(value);
            }

            public KeyValuePair<ColumnRole, string> Bind(string name)
            {
                return new KeyValuePair<ColumnRole, string>(this, name);
            }
        }

        public static KeyValuePair<ColumnRole, string> CreatePair(ColumnRole role, string name)
        {
            return new KeyValuePair<ColumnRole, string>(role, name);
        }

        /// <summary>
        /// The source ISchema.
        /// </summary>
        public readonly ISchema Schema;

        /// <summary>
        /// The Feature column, when there is exactly one (null otherwise).
        /// </summary>
        public readonly ColumnInfo Feature;

        /// <summary>
        /// The Label column, when there is exactly one (null otherwise).
        /// </summary>
        public readonly ColumnInfo Label;

        /// <summary>
        /// The Group column, when there is exactly one (null otherwise).
        /// </summary>
        public readonly ColumnInfo Group;

        /// <summary>
        /// The Weight column, when there is exactly one (null otherwise).
        /// </summary>
        public readonly ColumnInfo Weight;

        /// <summary>
        /// The Name column, when there is exactly one (null otherwise).
        /// </summary>
        public readonly ColumnInfo Name;

        // Maps from role to the associated column infos.
        private readonly Dictionary<string, IReadOnlyList<ColumnInfo>> _map;

        private RoleMappedSchema(ISchema schema, Dictionary<string, IReadOnlyList<ColumnInfo>> map)
        {
            Contracts.AssertValue(schema);
            Contracts.AssertValue(map);

            Schema = schema;
            _map = map;
            foreach (var kvp in _map)
            {
                Contracts.Assert(Utils.Size(kvp.Value) > 0);
                var cols = kvp.Value;
#if DEBUG
                foreach (var info in cols)
                    Contracts.Assert(!schema.IsHidden(info.Index), "How did a hidden column sneak in?");
#endif
                if (cols.Count == 1)
                {
                    switch (kvp.Key)
                    {
                    case FeatureString:
                        Feature = cols[0];
                        break;
                    case LabelString:
                        Label = cols[0];
                        break;
                    case GroupString:
                        Group = cols[0];
                        break;
                    case WeightString:
                        Weight = cols[0];
                        break;
                    case NameString:
                        Name = cols[0];
                        break;
                    }
                }
            }
        }

        private RoleMappedSchema(ISchema schema, Dictionary<string, List<ColumnInfo>> map)
            : this(schema, Copy(map))
        {
        }

        private static void Add(Dictionary<string, List<ColumnInfo>> map, ColumnRole role, ColumnInfo info)
        {
            Contracts.AssertValue(map);
            Contracts.AssertNonEmpty(role.Value);
            Contracts.AssertValue(info);

            List<ColumnInfo> list;
            if (!map.TryGetValue(role.Value, out list))
            {
                list = new List<ColumnInfo>();
                map.Add(role.Value, list);
            }
            list.Add(info);
        }

        private static Dictionary<string, List<ColumnInfo>> MapFromNames(ISchema schema, IEnumerable<KeyValuePair<ColumnRole, string>> roles)
        {
            Contracts.AssertValue(schema);
            Contracts.AssertValue(roles);

            var map = new Dictionary<string, List<ColumnInfo>>();
            foreach (var kvp in roles)
            {
                Contracts.CheckNonEmpty(kvp.Key.Value, nameof(roles), "Bad column role");
                if (string.IsNullOrEmpty(kvp.Value))
                    continue;
                var info = ColumnInfo.CreateFromName(schema, kvp.Value, kvp.Key.Value);
                Add(map, kvp.Key.Value, info);
            }
            return map;
        }

        private static Dictionary<string, List<ColumnInfo>> MapFromNamesOpt(ISchema schema, IEnumerable<KeyValuePair<ColumnRole, string>> roles)
        {
            Contracts.AssertValue(schema);
            Contracts.AssertValue(roles);

            var map = new Dictionary<string, List<ColumnInfo>>();
            foreach (var kvp in roles)
            {
                Contracts.CheckNonEmpty(kvp.Key.Value, nameof(roles), "Bad column role");
                if (string.IsNullOrEmpty(kvp.Value))
                    continue;
                ColumnInfo info;
                if (!ColumnInfo.TryCreateFromName(schema, kvp.Value, out info))
                    continue;
                Add(map, kvp.Key.Value, info);
            }
            return map;
        }

        /// <summary>
        /// Returns whether there are any columns with the given column role.
        /// </summary>
        public bool Has(ColumnRole role)
        {
            return role.Value != null && _map.ContainsKey(role.Value);
        }

        /// <summary>
        /// Returns whether there is exactly one column of the given role.
        /// </summary>
        public bool HasUnique(ColumnRole role)
        {
            IReadOnlyList<ColumnInfo> cols;
            return role.Value != null && _map.TryGetValue(role.Value, out cols) && cols.Count == 1;
        }

        /// <summary>
        /// Returns whether there are two or more columns of the given role.
        /// </summary>
        public bool HasMultiple(ColumnRole role)
        {
            IReadOnlyList<ColumnInfo> cols;
            return role.Value != null && _map.TryGetValue(role.Value, out cols) && cols.Count > 1;
        }

        /// <summary>
        /// If there are columns of the given role, this returns the infos as a readonly list. Otherwise,
        /// it returns null.
        /// </summary>
        public IReadOnlyList<ColumnInfo> GetColumns(ColumnRole role)
        {
            IReadOnlyList<ColumnInfo> list;
            if (role.Value != null && _map.TryGetValue(role.Value, out list))
                return list;
            return null;
        }

        /// <summary>
        /// An enumerable over all role-column associations within this object.
        /// </summary>
        public IEnumerable<KeyValuePair<ColumnRole, ColumnInfo>> GetColumnRoles()
        {
            foreach (var roleAndList in _map)
            {
                foreach (var info in roleAndList.Value)
                    yield return new KeyValuePair<ColumnRole, ColumnInfo>(roleAndList.Key, info);
            }
        }

        /// <summary>
        /// An enumerable over all role-column associations within this object.
        /// </summary>
        public IEnumerable<KeyValuePair<ColumnRole, string>> GetColumnRoleNames()
        {
            foreach (var roleAndList in _map)
            {
                foreach (var info in roleAndList.Value)
                    yield return new KeyValuePair<ColumnRole, string>(roleAndList.Key, info.Name);
            }
        }

        /// <summary>
        /// An enumerable over all role-column associations for the given role. This is a helper function
        /// for implementing the <see cref="ISchemaBoundMapper.GetInputColumnRoles"/> method.
        /// </summary>
        public IEnumerable<KeyValuePair<ColumnRole, string>> GetColumnRoleNames(ColumnRole role)
        {
            IReadOnlyList<ColumnInfo> list;
            if (role.Value != null && _map.TryGetValue(role.Value, out list))
            {
                foreach (var info in list)
                    yield return new KeyValuePair<ColumnRole, string>(role, info.Name);
            }
        }

        /// <summary>
        /// Returns the <see cref="ColumnInfo"/> corresponding to <paramref name="role"/> if there is
        /// exactly one such mapping, and otherwise throws an exception.
        /// </summary>
        /// <param name="role">The role to look up</param>
        /// <returns>The info corresponding to that role, assuming there was only one column
        /// mapped to that</returns>
        public ColumnInfo GetUniqueColumn(ColumnRole role)
        {
            var infos = GetColumns(role);
            if (Utils.Size(infos) != 1)
                throw Contracts.Except("Expected exactly one column with role '{0}', but found {1}.", role.Value, Utils.Size(infos));
            return infos[0];
        }

        private static Dictionary<string, IReadOnlyList<ColumnInfo>> Copy(Dictionary<string, List<ColumnInfo>> map)
        {
            var copy = new Dictionary<string, IReadOnlyList<ColumnInfo>>(map.Count);
            foreach (var kvp in map)
            {
                Contracts.Assert(Utils.Size(kvp.Value) > 0);
                var cols = kvp.Value.ToArray();
                copy.Add(kvp.Key, cols);
            }
            return copy;
        }

        /// <summary>
        /// Creates a RoleMappedSchema from the given schema with no column role assignments.
        /// </summary>
        public static RoleMappedSchema Create(ISchema schema)
        {
            Contracts.CheckValue(schema, nameof(schema));
            return new RoleMappedSchema(schema, new Dictionary<string, List<ColumnInfo>>());
        }

        /// <summary>
        /// Creates a RoleMappedSchema from the given schema and role/column-name pairs.
        /// This skips null or empty column-names.
        /// </summary>
        public static RoleMappedSchema Create(ISchema schema, params KeyValuePair<ColumnRole, string>[] roles)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(roles, nameof(roles));
            return new RoleMappedSchema(schema, MapFromNames(schema, roles));
        }

        /// <summary>
        /// Creates a RoleMappedSchema from the given schema and role/column-name pairs.
        /// This skips null or empty column-names.
        /// </summary>
        public static RoleMappedSchema Create(ISchema schema, IEnumerable<KeyValuePair<ColumnRole, string>> roles)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(roles, nameof(roles));
            return new RoleMappedSchema(schema, MapFromNames(schema, roles));
        }

        /// <summary>
        /// Creates a RoleMappedSchema from the given schema and role/column-name pairs.
        /// This skips null or empty column-names, or column-names that are not found in the schema.
        /// </summary>
        public static RoleMappedSchema CreateOpt(ISchema schema, IEnumerable<KeyValuePair<ColumnRole, string>> roles)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(roles, nameof(roles));
            return new RoleMappedSchema(schema, MapFromNamesOpt(schema, roles));
        }
    }

    /// <summary>
    /// Encapsulates an <see cref="IDataView"/> plus a corresponding <see cref="RoleMappedSchema"/>.
    /// Note that the schema of <see cref="RoleMappedSchema.Schema"/> of <see cref="Schema"/> is
    /// guaranteed to equal the the <see cref="ISchematized.Schema"/> of <see cref="Data"/>.
    /// </summary>
    public sealed class RoleMappedData
    {
        /// <summary>
        /// The data.
        /// </summary>
        public readonly IDataView Data;

        /// <summary>
        /// The role mapped schema. Note that Schema.Schema is guaranteed to be the same as Data.Schema.
        /// </summary>
        public readonly RoleMappedSchema Schema;

        private RoleMappedData(IDataView data, RoleMappedSchema schema)
        {
            Contracts.AssertValue(data);
            Contracts.AssertValue(schema);
            Contracts.Assert(schema.Schema == data.Schema);
            Data = data;
            Schema = schema;
        }

        /// <summary>
        /// Creates a RoleMappedData from the given data with no column role assignments.
        /// </summary>
        public static RoleMappedData Create(IDataView data)
        {
            Contracts.CheckValue(data, nameof(data));
            return new RoleMappedData(data, RoleMappedSchema.Create(data.Schema));
        }

        /// <summary>
        /// Creates a RoleMappedData from the given schema and role/column-name pairs.
        /// This skips null or empty column-names.
        /// </summary>
        public static RoleMappedData Create(IDataView data, params KeyValuePair<RoleMappedSchema.ColumnRole, string>[] roles)
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(roles, nameof(roles));
            return new RoleMappedData(data, RoleMappedSchema.Create(data.Schema, roles));
        }

        /// <summary>
        /// Creates a RoleMappedData from the given schema and role/column-name pairs.
        /// This skips null or empty column-names.
        /// </summary>
        public static RoleMappedData Create(IDataView data, IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> roles)
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(roles, nameof(roles));
            return new RoleMappedData(data, RoleMappedSchema.Create(data.Schema, roles));
        }

        /// <summary>
        /// Creates a RoleMappedData from the given schema and role/column-name pairs.
        /// This skips null or empty column-names, or column-names that are not found in the schema.
        /// </summary>
        public static RoleMappedData CreateOpt(IDataView data, IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> roles)
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(roles, nameof(roles));
            return new RoleMappedData(data, RoleMappedSchema.CreateOpt(data.Schema, roles));
        }
    }
}