// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This contains information about a column in an <see cref="IDataView"/>. It is essentially a convenience cache
    /// containing the name, column index, and column type for the column. The intended usage is that users of <see cref="RoleMappedSchema"/>
    /// will have a convenient method of getting the index and type without having to separately query it through the <see cref="ISchema"/>,
    /// since practically the first thing a consumer of a <see cref="RoleMappedSchema"/> will want to do once they get a mappping is get
    /// the type and index of the corresponding column.
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
            if (!TryCreateFromName(schema, name, out var colInfo))
                throw Contracts.ExceptParam(nameof(name), $"{descName} column '{name}' not found");

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
            if (!schema.TryGetColumnIndex(name, out int index))
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
    /// cref="Label"/>), but can hold an arbitrary set of column infos. The convenience fields are non-null if and only
    /// if there is a unique column with the corresponding role. When there are no such columns or more than one such
    /// column, the field is <c>null</c>. The <see cref="Has"/>, <see cref="HasUnique"/>, and <see cref="HasMultiple"/>
    /// methods provide some cardinality information. Note that all columns assigned roles are guaranteed to be non-hidden
    /// in this schema.
    /// </summary>
    /// <remarks>
    /// Note that instances of this class are, like instances of <see cref="ISchema"/>, immutable.
    /// 
    /// It is often the case that one wishes to bundle the actual data with the role mappings, not just the schema. For
    /// that case, please use the <see cref="RoleMappedData"/> class.
    /// 
    /// Note that there is no need for components consuming a <see cref="RoleMappedData"/> or <see cref="RoleMappedSchema"/>
    /// to make use of every defined mapping. Consuming components are also expected to ignore any <see cref="ColumnRole"/>
    /// they do not handle. They may very well however complain if a mapping they wanted to see is not present, or the column(s)
    /// mapped from the role are not of the form they require.
    /// </remarks>
    /// <seealso cref="ColumnRole"/>
    /// <seealso cref="RoleMappedData"/>
    public sealed class RoleMappedSchema
    {
        private const string FeatureString = "Feature";
        private const string LabelString = "Label";
        private const string GroupString = "Group";
        private const string WeightString = "Weight";
        private const string NameString = "Name";
        private const string FeatureContributionsString = "FeatureContributions";

        /// <summary>
        /// Instances of this are the keys of a <see cref="RoleMappedSchema"/>. This class also holds some important
        /// commonly used pre-defined instances available (e.g., <see cref="Label"/>, <see cref="Feature"/>) that should
        /// be used when possible for consistency reasons. However, practitioners should not be afraid to declare custom
        /// roles if approppriate for their task.
        /// </summary>
        public struct ColumnRole
        {
            /// <summary>
            /// Role for features. Commonly used as the independent variables given to trainers, and scorers.
            /// </summary>
            public static ColumnRole Feature => FeatureString;

            /// <summary>
            /// Role for labels. Commonly used as the dependent variables given to trainers, and evaluators.
            /// </summary>
            public static ColumnRole Label => LabelString;

            /// <summary>
            /// Role for group ID. Commonly used in ranking applications, for defining query boundaries, or
            /// sequence classification, for defining the boundaries of an utterance.
            /// </summary>
            public static ColumnRole Group => GroupString;

            /// <summary>
            /// Role for sample weights. Commonly used to point to a number to make trainers give more weight
            /// to a particular example.
            /// </summary>
            public static ColumnRole Weight => WeightString;

            /// <summary>
            /// Role for sample names. Useful for informational and tracking purposes when scoring, but typically
            /// without affecting results.
            /// </summary>
            public static ColumnRole Name => NameString;

            // REVIEW: Does this really belong here?
            /// <summary>
            /// Role for feature contributions. Useful for specific diagnostic functionality.
            /// </summary>
            public static ColumnRole FeatureContributions => FeatureContributionsString;

            /// <summary>
            /// The string value for the role. Guaranteed to be non-empty.
            /// </summary>
            public readonly string Value;

            /// <summary>
            /// Constructor for the column role.
            /// </summary>
            /// <param name="value">The value for the role. Must be non-empty.</param>
            public ColumnRole(string value)
            {
                Contracts.CheckNonEmpty(value, nameof(value));
                Value = value;
            }

            public static implicit operator ColumnRole(string value)
                => new ColumnRole(value);

            /// <summary>
            /// Convenience method for creating a mapping pair from a role to a column name
            /// for giving to constructors of <see cref="RoleMappedSchema"/> and <see cref="RoleMappedData"/>.
            /// </summary>
            /// <param name="name">The column name to map to. Can be <c>null</c>, in which case when used
            /// to construct a role mapping structure this pair will be ignored</param>
            /// <returns>A key-value pair with this instance as the key and <paramref name="name"/> as the value</returns>
            public KeyValuePair<ColumnRole, string> Bind(string name)
                => new KeyValuePair<ColumnRole, string>(this, name);
        }

        public static KeyValuePair<ColumnRole, string> CreatePair(ColumnRole role, string name)
            => new KeyValuePair<ColumnRole, string>(role, name);

        /// <summary>
        /// The source <see cref="ISchema"/>.
        /// </summary>
        public ISchema Schema { get; }

        /// <summary>
        /// The <see cref="ColumnRole.Feature"/> column, when there is exactly one (null otherwise).
        /// </summary>
        public ColumnInfo Feature { get; }

        /// <summary>
        /// The <see cref="ColumnRole.Label"/> column, when there is exactly one (null otherwise).
        /// </summary>
        public ColumnInfo Label { get; }

        /// <summary>
        /// The <see cref="ColumnRole.Group"/> column, when there is exactly one (null otherwise).
        /// </summary>
        public ColumnInfo Group { get; }

        /// <summary>
        /// The <see cref="ColumnRole.Weight"/> column, when there is exactly one (null otherwise).
        /// </summary>
        public ColumnInfo Weight { get; }

        /// <summary>
        /// The <see cref="ColumnRole.Name"/> column, when there is exactly one (null otherwise).
        /// </summary>
        public ColumnInfo Name { get; }

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

            if (!map.TryGetValue(role.Value, out var list))
            {
                list = new List<ColumnInfo>();
                map.Add(role.Value, list);
            }
            list.Add(info);
        }

        private static Dictionary<string, List<ColumnInfo>> MapFromNames(ISchema schema, IEnumerable<KeyValuePair<ColumnRole, string>> roles, bool opt = false)
        {
            Contracts.AssertValue(schema);
            Contracts.AssertValue(roles);

            var map = new Dictionary<string, List<ColumnInfo>>();
            foreach (var kvp in roles)
            {
                Contracts.AssertNonEmpty(kvp.Key.Value);
                if (string.IsNullOrEmpty(kvp.Value))
                    continue;
                ColumnInfo info;
                if (!opt)
                    info = ColumnInfo.CreateFromName(schema, kvp.Value, kvp.Key.Value);
                else if (!ColumnInfo.TryCreateFromName(schema, kvp.Value, out info))
                    continue;
                Add(map, kvp.Key.Value, info);
            }
            return map;
        }

        /// <summary>
        /// Returns whether there are any columns with the given column role.
        /// </summary>
        public bool Has(ColumnRole role)
            => _map.ContainsKey(role.Value);

        /// <summary>
        /// Returns whether there is exactly one column of the given role.
        /// </summary>
        public bool HasUnique(ColumnRole role)
            => _map.TryGetValue(role.Value, out var cols) && cols.Count == 1;

        /// <summary>
        /// Returns whether there are two or more columns of the given role.
        /// </summary>
        public bool HasMultiple(ColumnRole role)
            => _map.TryGetValue(role.Value, out var cols) && cols.Count > 1;

        /// <summary>
        /// If there are columns of the given role, this returns the infos as a readonly list. Otherwise,
        /// it returns null.
        /// </summary>
        public IReadOnlyList<ColumnInfo> GetColumns(ColumnRole role)
            => _map.TryGetValue(role.Value, out var list) ? list : null;

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
            if (_map.TryGetValue(role.Value, out var list))
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
        /// Constructor given a schema, and mapping pairs of roles to columns in the schema.
        /// This skips null or empty column-names. It will also skip column-names that are not
        /// found in the schema if <paramref name="opt"/> is true.
        /// </summary>
        /// <param name="schema">The schema over which roles are defined</param>
        /// <param name="opt">Whether to consider the column names specified "optional" or not. If <c>false</c> then any non-empty
        /// values for the column names that does not appear in <paramref name="schema"/> will result in an exception being thrown,
        /// but if <c>true</c> such values will be ignored</param>
        /// <param name="roles">The column role to column name mappings</param>
        public RoleMappedSchema(ISchema schema, bool opt = false, params KeyValuePair<ColumnRole, string>[] roles)
            : this(Contracts.CheckRef(schema, nameof(schema)), Contracts.CheckRef(roles, nameof(roles)), opt)
        {
        }

        /// <summary>
        /// Constructor given a schema, and mapping pairs of roles to columns in the schema.
        /// This skips null or empty column names. It will also skip column-names that are not
        /// found in the schema if <paramref name="opt"/> is true.
        /// </summary>
        /// <param name="schema">The schema over which roles are defined</param>
        /// <param name="roles">The column role to column name mappings</param>
        /// <param name="opt">Whether to consider the column names specified "optional" or not. If <c>false</c> then any non-empty
        /// values for the column names that does not appear in <paramref name="schema"/> will result in an exception being thrown,
        /// but if <c>true</c> such values will be ignored</param>
        public RoleMappedSchema(ISchema schema, IEnumerable<KeyValuePair<ColumnRole, string>> roles, bool opt = false)
            : this(Contracts.CheckRef(schema, nameof(schema)),
                  MapFromNames(schema, Contracts.CheckRef(roles, nameof(roles)), opt))
        {
        }

        private static IEnumerable<KeyValuePair<ColumnRole, string>> PredefinedRolesHelper(
            string label, string feature, string group, string weight, string name,
            IEnumerable<KeyValuePair<ColumnRole, string>> custom = null)
        {
            if (!string.IsNullOrWhiteSpace(label))
                yield return ColumnRole.Label.Bind(label);
            if (!string.IsNullOrWhiteSpace(feature))
                yield return ColumnRole.Feature.Bind(feature);
            if (!string.IsNullOrWhiteSpace(group))
                yield return ColumnRole.Group.Bind(group);
            if (!string.IsNullOrWhiteSpace(weight))
                yield return ColumnRole.Weight.Bind(weight);
            if (!string.IsNullOrWhiteSpace(name))
                yield return ColumnRole.Name.Bind(name);
            if (custom != null)
            {
                foreach (var role in custom)
                    yield return role;
            }
        }

        /// <summary>
        /// Convenience constructor for role-mappings over the commonly used roles. Note that if any column name specified
        /// is <c>null</c> or whitespace, it is ignored.
        /// </summary>
        /// <param name="schema">The schema over which roles are defined</param>
        /// <param name="label">The column name that will be mapped to the <see cref="ColumnRole.Label"/> role</param>
        /// <param name="feature">The column name that will be mapped to the <see cref="ColumnRole.Feature"/> role</param>
        /// <param name="group">The column name that will be mapped to the <see cref="ColumnRole.Group"/> role</param>
        /// <param name="weight">The column name that will be mapped to the <see cref="ColumnRole.Weight"/> role</param>
        /// <param name="name">The column name that will be mapped to the <see cref="ColumnRole.Name"/> role</param>
        /// <param name="custom">Any additional desired custom column role mappings</param>
        /// <param name="opt">Whether to consider the column names specified "optional" or not. If <c>false</c> then any non-empty
        /// values for the column names that does not appear in <paramref name="schema"/> will result in an exception being thrown,
        /// but if <c>true</c> such values will be ignored</param>
        public RoleMappedSchema(ISchema schema, string label, string feature,
            string group = null, string weight = null, string name = null,
            IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> custom = null, bool opt = false)
            : this(Contracts.CheckRef(schema, nameof(schema)), PredefinedRolesHelper(label, feature, group, weight, name, custom), opt)
        {
            Contracts.CheckValueOrNull(label);
            Contracts.CheckValueOrNull(feature);
            Contracts.CheckValueOrNull(group);
            Contracts.CheckValueOrNull(weight);
            Contracts.CheckValueOrNull(name);
            Contracts.CheckValueOrNull(custom);
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
        public IDataView Data { get; }

        /// <summary>
        /// The role mapped schema. Note that <see cref="Schema"/>'s <see cref="RoleMappedSchema.Schema"/> is
        /// guaranteed to be the same as <see cref="Data"/>'s <see cref="ISchematized.Schema"/>.
        /// </summary>
        public RoleMappedSchema Schema { get; }

        private RoleMappedData(IDataView data, RoleMappedSchema schema)
        {
            Contracts.AssertValue(data);
            Contracts.AssertValue(schema);
            Contracts.Assert(schema.Schema == data.Schema);
            Data = data;
            Schema = schema;
        }

        /// <summary>
        /// Constructor given a data view, and mapping pairs of roles to columns in the data view's schema.
        /// This skips null or empty column-names. It will also skip column-names that are not
        /// found in the schema if <paramref name="opt"/> is true.
        /// </summary>
        /// <param name="data">The data over which roles are defined</param>
        /// <param name="opt">Whether to consider the column names specified "optional" or not. If <c>false</c> then any non-empty
        /// values for the column names that does not appear in <paramref name="data"/>'s schema will result in an exception being thrown,
        /// but if <c>true</c> such values will be ignored</param>
        /// <param name="roles">The column role to column name mappings</param>
        public RoleMappedData(IDataView data, bool opt = false, params KeyValuePair<RoleMappedSchema.ColumnRole, string>[] roles)
            : this(Contracts.CheckRef(data, nameof(data)), new RoleMappedSchema(data.Schema, Contracts.CheckRef(roles, nameof(roles)), opt))
        {
        }

        /// <summary>
        /// Constructor given a data view, and mapping pairs of roles to columns in the data view's schema.
        /// This skips null or empty column-names. It will also skip column-names that are not
        /// found in the schema if <paramref name="opt"/> is true.
        /// </summary>
        /// <param name="data">The schema over which roles are defined</param>
        /// <param name="roles">The column role to column name mappings</param>
        /// <param name="opt">Whether to consider the column names specified "optional" or not. If <c>false</c> then any non-empty
        /// values for the column names that does not appear in <paramref name="data"/>'s schema will result in an exception being thrown,
        /// but if <c>true</c> such values will be ignored</param>
        public RoleMappedData(IDataView data, IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> roles, bool opt = false)
            : this(Contracts.CheckRef(data, nameof(data)), new RoleMappedSchema(data.Schema, Contracts.CheckRef(roles, nameof(roles)), opt))
        {
        }

        /// <summary>
        /// Convenience constructor for role-mappings over the commonly used roles. Note that if any column name specified
        /// is <c>null</c> or whitespace, it is ignored.
        /// </summary>
        /// <param name="data">The data over which roles are defined</param>
        /// <param name="label">The column name that will be mapped to the <see cref="RoleMappedSchema.ColumnRole.Label"/> role</param>
        /// <param name="feature">The column name that will be mapped to the <see cref="RoleMappedSchema.ColumnRole.Feature"/> role</param>
        /// <param name="group">The column name that will be mapped to the <see cref="RoleMappedSchema.ColumnRole.Group"/> role</param>
        /// <param name="weight">The column name that will be mapped to the <see cref="RoleMappedSchema.ColumnRole.Weight"/> role</param>
        /// <param name="name">The column name that will be mapped to the <see cref="RoleMappedSchema.ColumnRole.Name"/> role</param>
        /// <param name="custom">Any additional desired custom column role mappings</param>
        /// <param name="opt">Whether to consider the column names specified "optional" or not. If <c>false</c> then any non-empty
        /// values for the column names that does not appear in <paramref name="data"/>'s schema will result in an exception being thrown,
        /// but if <c>true</c> such values will be ignored</param>
        public RoleMappedData(IDataView data, string label, string feature,
            string group = null, string weight = null, string name = null,
            IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> custom = null, bool opt = false)
            : this(Contracts.CheckRef(data, nameof(data)),
                  new RoleMappedSchema(data.Schema, label, feature, group, weight, name, custom, opt))
        {
            Contracts.CheckValueOrNull(label);
            Contracts.CheckValueOrNull(feature);
            Contracts.CheckValueOrNull(group);
            Contracts.CheckValueOrNull(weight);
            Contracts.CheckValueOrNull(name);
            Contracts.CheckValueOrNull(custom);
        }
    }
}