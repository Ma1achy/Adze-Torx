# G2 — learned commitment

PHASE_G_GENERATED_STRUCTURE_FUNCTIONAL

```json
{
  "identity": "5cb5e7363b14239b4c65a7dda1fb70b569e0b92a935e1f2c67311496a15f1d99",
  "selections": {
    "A": {
      "selected": {
        "step": 1000,
        "path": "results/runs/phase_g/g23/v1/seed0/A/snapshots/update_01000.pkl",
        "sha256": "6a82112b480840ffe2f759fd5180d8e733c6ee02e7cd89543e310babbde9b9d0",
        "mode": "teacher",
        "model_tree": "24e90f4d7df68905ecda08d4c228b647cc629a7e630d914312e40f2b53f62a03",
        "optimizer_tree": "3f35f444f4e91b15a6970d1041a80569191ca90ef19f2d8c4d4250a0139f1d85"
      },
      "candidates": [
        {
          "step": 0,
          "eligible": true,
          "query_loss": 1.2422391176223755,
          "query_accuracy": 0.5983186960220337,
          "model_tree": "94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2",
          "optimizer_tree": "00e123f3c21c52cdf6ff306359e33b1b84544a49ec5e34760ca39c9b9bcc79eb"
        },
        {
          "step": 250,
          "eligible": true,
          "query_loss": 0.06991052627563477,
          "query_accuracy": 0.9747806787490845,
          "model_tree": "d831f74fc2582d4d59ae84480245dc7a000607d5951b60baee6dbe9455d02127",
          "optimizer_tree": "a7605b2651dd1de10ce527acff48579d40c9b87bfd92ff10462753dfc10f9f7b"
        },
        {
          "step": 500,
          "eligible": true,
          "query_loss": 0.023316137492656708,
          "query_accuracy": 0.9934210777282715,
          "model_tree": "b164f95d2c4e82d293cd84e46d6b6ecf5c444090be8dbd576f193b3a0fafb4b7",
          "optimizer_tree": "5a7ffe250c09d33c824b827b8dcb4c1f27d281bcace4c01d85cdc775d6866438"
        },
        {
          "step": 1000,
          "eligible": true,
          "query_loss": 0.012281790375709534,
          "query_accuracy": 0.995248556137085,
          "model_tree": "24e90f4d7df68905ecda08d4c228b647cc629a7e630d914312e40f2b53f62a03",
          "optimizer_tree": "3f35f444f4e91b15a6970d1041a80569191ca90ef19f2d8c4d4250a0139f1d85"
        }
      ]
    },
    "B": {
      "selected": {
        "step": 1000,
        "path": "results/runs/phase_g/g23/v1/seed0/B/snapshots/update_01000.pkl",
        "sha256": "ff344d72489c0923cc014e40b2d5b647f04c8c6e4af928713019838da07a6446",
        "mode": "argmax",
        "model_tree": "a3ef451d2842f4d7481f8c705d6754912ca187b1f2bbc63665baa85c7e55bd6c",
        "optimizer_tree": "9f445e86ba2f73f98a89d5c68d3b96deceec75f5b929271b949cb0f3a3c36c75"
      },
      "candidates": [
        {
          "step": 0,
          "eligible": true,
          "query_loss": 1.2521973848342896,
          "query_accuracy": 0.5975877046585083,
          "model_tree": "94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2",
          "optimizer_tree": "00e123f3c21c52cdf6ff306359e33b1b84544a49ec5e34760ca39c9b9bcc79eb"
        },
        {
          "step": 250,
          "eligible": true,
          "query_loss": 0.09607461094856262,
          "query_accuracy": 0.9689327478408813,
          "model_tree": "380d4baa0e3d2e1b8d9ab882203d9a3a82c1cf7beb9346e882c09e094cd0ec06",
          "optimizer_tree": "7113886e0e6aa55255a72ff126ad678bbba8e5e239458ccd123903e3d0c488b4"
        },
        {
          "step": 500,
          "eligible": true,
          "query_loss": 0.03075006976723671,
          "query_accuracy": 0.9894006252288818,
          "model_tree": "31410cc3dc321280c357a5809bc7178f1b1f7cee70b0f7d0947e046be75318a8",
          "optimizer_tree": "87904dccdeb7d3aff01638b126eb150c83379021ad25e19a77b86ef4e5eabab2"
        },
        {
          "step": 1000,
          "eligible": true,
          "query_loss": 0.0198395736515522,
          "query_accuracy": 0.9945175647735596,
          "model_tree": "a3ef451d2842f4d7481f8c705d6754912ca187b1f2bbc63665baa85c7e55bd6c",
          "optimizer_tree": "9f445e86ba2f73f98a89d5c68d3b96deceec75f5b929271b949cb0f3a3c36c75"
        }
      ]
    },
    "C": {
      "selected": {
        "step": 1000,
        "path": "results/runs/phase_g/g23/v1/seed0/C/snapshots/update_01000.pkl",
        "sha256": "b4896a4d44bc74928c68824bd2be2c52f726eac1f0675a14f77b54d123a7abc8",
        "mode": "sample",
        "model_tree": "20881b8be95af4d811fb2ce4531893409c4d6f2a2955269d53fa07d660e1384f",
        "optimizer_tree": "974ffaad4b13b146d0d5a8f9f1c222fcb54bd354b7f5fadb469d1ada5bcb5268"
      },
      "candidates": [
        {
          "step": 0,
          "eligible": false,
          "query_loss": 1.752639889717102,
          "query_accuracy": 0.5760234594345093,
          "model_tree": "94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2",
          "optimizer_tree": "00e123f3c21c52cdf6ff306359e33b1b84544a49ec5e34760ca39c9b9bcc79eb"
        },
        {
          "step": 250,
          "eligible": true,
          "query_loss": 0.2026842087507248,
          "query_accuracy": 0.9555007815361023,
          "model_tree": "96b0eabcc3514bb075e47df7f971c72355b4afca34c138c2805b713e59509d10",
          "optimizer_tree": "f1930fb6579074598b16e329fd4b720eb36f9bc54ec0cc61d53a85f6d0b59d46"
        },
        {
          "step": 500,
          "eligible": true,
          "query_loss": 0.05632421001791954,
          "query_accuracy": 0.9897660613059998,
          "model_tree": "1a276b3acd119941d36e13c89c80b68f48805644c0ab85d10fea2ecb63ddaa19",
          "optimizer_tree": "1416f22f8aef9713dd9e254f7a0ff96d28c23db92af3c7d8953eddbe988aa250"
        },
        {
          "step": 1000,
          "eligible": true,
          "query_loss": 0.03751153126358986,
          "query_accuracy": 0.9904970526695251,
          "model_tree": "20881b8be95af4d811fb2ce4531893409c4d6f2a2955269d53fa07d660e1384f",
          "optimizer_tree": "974ffaad4b13b146d0d5a8f9f1c222fcb54bd354b7f5fadb469d1ada5bcb5268"
        }
      ]
    }
  },
  "modes": {
    "B:argmax": {
      "passed": true,
      "checks": {
        "boundary": true,
        "length": true,
        "valid": true,
        "reach": true,
        "accuracy": true,
        "query_loss": true,
        "nll": true,
        "nll_support": true,
        "uncertainty": true,
        "competence": true,
        "chance_margin": true
      },
      "summary": {
        "b_accuracy": 1.0,
        "b_ce": 0.0003224200918339193,
        "b_fn": 0.0,
        "b_fp": 0.0,
        "b_support": 9.368421052631579,
        "b_tp": 4.93640350877193,
        "byte_accuracy": 0.9981886148452759,
        "byte_exact": 0.930921052631579,
        "byte_loss": 0.005451111122965813,
        "byte_nll": 0.005451111076364517,
        "byte_nll_support": 36.0,
        "commit_change": 0.8626644611358643,
        "descendant": 0.051755912601947784,
        "ell_ce": 5.525889355340041e-05,
        "emitted_length": 36.0,
        "entropy": 0.0640563815832138,
        "extent_accuracy": 1.0,
        "extent_exact": 1.0,
        "extra": 0.0,
        "h_mse": 0.05293045565485954,
        "length_accuracy": 1.0,
        "log_prob": 0.0,
        "loss_per_example": 0.05213358998298645,
        "missing": 0.0,
        "missing_query": 0.0,
        "nonquery_accuracy": 0.9983263611793518,
        "nonquery_loss": 0.004720069468021393,
        "nonquery_nll": 0.004720069671552533,
        "nonquery_nll_support": 32.0,
        "partial_correct": 0.7368421052631579,
        "partial_support": 0.7368421052631579,
        "peer_reach": 1.0,
        "prediction_b_accuracy": 1.0,
        "prediction_extent_accuracy": 1.0,
        "prediction_extent_exact": 1.0,
        "query_accuracy": 0.9945175647735596,
        "query_exact": 0.9835526315789473,
        "query_loss": 0.0198395736515522,
        "query_nll": 0.01983957256160283,
        "query_nll_support": 3.0,
        "structure_exact": 1.0,
        "true_length": 36.0,
        "valid": 1.0,
        "active_boundary_f1": 1.0,
        "extent_class_support": [
          20640.0,
          224.0,
          224.0,
          224.0,
          7872.0
        ],
        "extent_class_recall": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ],
        "partial_extent_accuracy": 1.0
      },
      "cut": {
        "b_accuracy": 0.859247088432312,
        "b_ce": 0.0003224200918339193,
        "b_fn": 0.0,
        "b_fp": 1.0,
        "b_support": 9.368421052631579,
        "b_tp": 4.93640350877193,
        "byte_accuracy": 0.888109564781189,
        "byte_exact": 0.0,
        "byte_loss": 3.150386333465576,
        "byte_nll": 3.150386443132894,
        "byte_nll_support": 36.0,
        "commit_change": 0.8626644611358643,
        "descendant": 0.051755912601947784,
        "ell_ce": 5.525889355340041e-05,
        "emitted_length": 36.0,
        "entropy": 0.0640563815832138,
        "extent_accuracy": 1.0,
        "extent_exact": 1.0,
        "extra": 0.0,
        "h_mse": 0.06002635136246681,
        "length_accuracy": 1.0,
        "log_prob": 0.0,
        "loss_per_example": 0.05213358998298645,
        "missing": 0.0,
        "missing_query": 0.0,
        "nonquery_accuracy": 0.9979032278060913,
        "nonquery_loss": 0.005420289468020201,
        "nonquery_nll": 0.005420289907561581,
        "nonquery_nll_support": 32.0,
        "partial_correct": 0.7368421052631579,
        "partial_support": 0.7368421052631579,
        "peer_reach": 0.0,
        "prediction_b_accuracy": 1.0,
        "prediction_extent_accuracy": 1.0,
        "prediction_extent_exact": 1.0,
        "query_accuracy": 0.029970761388540268,
        "query_exact": 0.0,
        "query_loss": 28.61705780029297,
        "query_nll": 28.617057469983894,
        "query_nll_support": 3.0,
        "structure_exact": 0.0,
        "true_length": 36.0,
        "valid": 1.0,
        "base_log_prob": 0.0,
        "intervened": 1.0,
        "active_boundary_f1": 0.9080274304154902,
        "extent_class_support": [
          20640.0,
          224.0,
          224.0,
          224.0,
          7872.0
        ],
        "extent_class_recall": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ],
        "partial_extent_accuracy": 1.0
      },
      "paired": {
        "query_accuracy": {
          "delta": 0.9645467400550842,
          "ci95": [
            0.957602322101593,
            0.9711257815361023
          ],
          "relative": 32.182926177978516,
          "relative_ci95": [
            26.46439290046692,
            40.24280652999878
          ],
          "support": 912,
          "replicates": 2000
        },
        "query_loss": {
          "delta": -28.597219467163086,
          "ci95": [
            -29.320662832260133,
            -27.88202667236328
          ],
          "relative": -0.9993067383766174,
          "relative_ci95": [
            -0.999634464085102,
            -0.9989410921931267
          ],
          "support": 912,
          "replicates": 2000
        },
        "query_nll": {
          "delta": -28.597217897422293,
          "ci95": [
            -29.31204786092409,
            -27.866635564510837
          ],
          "relative": -0.9993067221330351,
          "relative_ci95": [
            -0.9996273493269722,
            -0.998911791615713
          ],
          "support": 912,
          "replicates": 2000
        }
      }
    },
    "C:argmax": {
      "passed": true,
      "checks": {
        "boundary": true,
        "length": true,
        "valid": true,
        "reach": true,
        "accuracy": true,
        "query_loss": true,
        "nll": true,
        "nll_support": true,
        "uncertainty": true,
        "competence": true,
        "chance_margin": true
      },
      "summary": {
        "b_accuracy": 1.0,
        "b_ce": 0.00023380503989756107,
        "b_fn": 0.0,
        "b_fp": 0.0,
        "b_support": 9.368421052631579,
        "b_tp": 4.93640350877193,
        "byte_accuracy": 0.9980664849281311,
        "byte_exact": 0.9221491228070176,
        "byte_loss": 0.0069854059256613255,
        "byte_nll": 0.006985406131882106,
        "byte_nll_support": 36.0,
        "commit_change": 0.8626644611358643,
        "descendant": 0.07728991657495499,
        "ell_ce": 5.236273864284158e-05,
        "emitted_length": 36.0,
        "entropy": 0.046696737408638,
        "extent_accuracy": 1.0,
        "extent_exact": 1.0,
        "extra": 0.0,
        "h_mse": 0.06544499099254608,
        "length_accuracy": 1.0,
        "log_prob": 0.0,
        "loss_per_example": 0.07757608592510223,
        "missing": 0.0,
        "missing_query": 0.0,
        "nonquery_accuracy": 0.9986688494682312,
        "nonquery_loss": 0.0044529554434120655,
        "nonquery_nll": 0.004452955433162638,
        "nonquery_nll_support": 32.0,
        "partial_correct": 0.7368421052631579,
        "partial_support": 0.7368421052631579,
        "peer_reach": 1.0,
        "prediction_b_accuracy": 1.0,
        "prediction_extent_accuracy": 1.0,
        "prediction_extent_exact": 1.0,
        "query_accuracy": 0.9904970526695251,
        "query_exact": 0.9736842105263158,
        "query_loss": 0.03758201748132706,
        "query_nll": 0.03758201653474216,
        "query_nll_support": 3.0,
        "structure_exact": 1.0,
        "true_length": 36.0,
        "valid": 1.0,
        "active_boundary_f1": 1.0,
        "extent_class_support": [
          20640.0,
          224.0,
          224.0,
          224.0,
          7872.0
        ],
        "extent_class_recall": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ],
        "partial_extent_accuracy": 1.0
      },
      "cut": {
        "b_accuracy": 0.859247088432312,
        "b_ce": 0.00023380503989756107,
        "b_fn": 0.0,
        "b_fp": 1.0,
        "b_support": 9.368421052631579,
        "b_tp": 4.93640350877193,
        "byte_accuracy": 0.8897231221199036,
        "byte_exact": 0.0,
        "byte_loss": 4.227571487426758,
        "byte_nll": 4.22757168816762,
        "byte_nll_support": 36.0,
        "commit_change": 0.8626644611358643,
        "descendant": 0.07728991657495499,
        "ell_ce": 5.236273864284158e-05,
        "emitted_length": 36.0,
        "entropy": 0.046696737408638,
        "extent_accuracy": 1.0,
        "extent_exact": 1.0,
        "extra": 0.0,
        "h_mse": 0.0735095664858818,
        "length_accuracy": 1.0,
        "log_prob": 0.0,
        "loss_per_example": 0.07757608592510223,
        "missing": 0.0,
        "missing_query": 0.0,
        "nonquery_accuracy": 0.9987908005714417,
        "nonquery_loss": 0.004401737358421087,
        "nonquery_nll": 0.004401737338152478,
        "nonquery_nll_support": 32.0,
        "partial_correct": 0.7368421052631579,
        "partial_support": 0.7368421052631579,
        "peer_reach": 0.0,
        "prediction_b_accuracy": 1.0,
        "prediction_extent_accuracy": 1.0,
        "prediction_extent_exact": 1.0,
        "query_accuracy": 0.03618421033024788,
        "query_exact": 0.0,
        "query_loss": 38.86946487426758,
        "query_nll": 38.869464684996686,
        "query_nll_support": 3.0,
        "structure_exact": 0.0,
        "true_length": 36.0,
        "valid": 1.0,
        "base_log_prob": 0.0,
        "intervened": 1.0,
        "active_boundary_f1": 0.9080274304154902,
        "extent_class_support": [
          20640.0,
          224.0,
          224.0,
          224.0,
          7872.0
        ],
        "extent_class_recall": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ],
        "partial_extent_accuracy": 1.0
      },
      "paired": {
        "query_accuracy": {
          "delta": 0.9543128609657288,
          "ci95": [
            0.9466373920440674,
            0.9616228342056274
          ],
          "relative": 26.373737335205078,
          "relative_ci95": [
            22.179271697998047,
            32.34814929962157
          ],
          "support": 912,
          "replicates": 2000
        },
        "query_loss": {
          "delta": -38.83188247680664,
          "ci95": [
            -39.87950639724731,
            -37.84472923278809
          ],
          "relative": -0.9990330934524536,
          "relative_ci95": [
            -0.9994551777839661,
            -0.9985390260815621
          ],
          "support": 912,
          "replicates": 2000
        },
        "query_nll": {
          "delta": -38.83188266846195,
          "ci95": [
            -39.83116590772687,
            -37.80634914679683
          ],
          "relative": -0.9990331223535155,
          "relative_ci95": [
            -0.9994445379840706,
            -0.9985180970381758
          ],
          "support": 912,
          "replicates": 2000
        }
      }
    },
    "C:sample": {
      "passed": true,
      "checks": {
        "boundary": true,
        "length": true,
        "valid": true,
        "reach": true,
        "accuracy": true,
        "query_loss": true,
        "nll": true,
        "nll_support": true,
        "uncertainty": true,
        "competence": true,
        "chance_margin": true
      },
      "summary": {
        "b_accuracy": 0.9995071887969971,
        "b_ce": 0.00023380503989756107,
        "b_fn": 0.004660087719298245,
        "b_fp": 0.0005482456140350877,
        "b_support": 9.368421052631579,
        "b_tp": 4.931743421052632,
        "byte_accuracy": 0.9979686141014099,
        "byte_exact": 0.9199561403508771,
        "byte_loss": 0.007525458466261625,
        "byte_nll": 0.006998778998666561,
        "byte_nll_support": 35.999451754385966,
        "commit_change": 0.8625017404556274,
        "descendant": 0.07775753736495972,
        "ell_ce": 5.236273864284158e-05,
        "emitted_length": 36.00137061403509,
        "entropy": 0.046696737408638,
        "extent_accuracy": 0.9999229311943054,
        "extent_exact": 0.9975328947368421,
        "extra": 0.001918859649122807,
        "h_mse": 0.0654410794377327,
        "length_accuracy": 0.9975328947368421,
        "log_prob": -0.04698963463306427,
        "loss_per_example": 0.07804370671510696,
        "missing": 0.0005482456140350877,
        "missing_query": 0.0,
        "nonquery_accuracy": 0.9985420107841492,
        "nonquery_loss": 0.0051884097047150135,
        "nonquery_nll": 0.004471952069680263,
        "nonquery_nll_support": 31.999451754385966,
        "partial_correct": 0.734375,
        "partial_support": 0.7368421052631579,
        "peer_reach": 1.0,
        "prediction_b_accuracy": 1.0,
        "prediction_extent_accuracy": 1.0,
        "prediction_extent_exact": 1.0,
        "query_accuracy": 0.9904970526695251,
        "query_exact": 0.9736842105263158,
        "query_loss": 0.03751153126358986,
        "query_nll": 0.03751153182419007,
        "query_nll_support": 3.0,
        "structure_exact": 0.9923245614035088,
        "true_length": 36.0,
        "valid": 1.0,
        "active_boundary_f1": 0.9994722368823088,
        "extent_class_support": [
          20640.0,
          224.0,
          224.0,
          224.0,
          7872.0
        ],
        "extent_class_recall": [
          1.0,
          0.9910714285714286,
          1.0,
          0.9988839285714286,
          1.0
        ],
        "partial_extent_accuracy": 0.9966517857142857
      },
      "cut": {
        "b_accuracy": 0.8588488101959229,
        "b_ce": 0.00023380503989756107,
        "b_fn": 0.003837719298245614,
        "b_fp": 1.000548245614035,
        "b_support": 9.368421052631579,
        "b_tp": 4.932565789473684,
        "byte_accuracy": 0.889634907245636,
        "byte_exact": 0.0,
        "byte_loss": 4.228038311004639,
        "byte_nll": 4.227559339859111,
        "byte_nll_support": 35.999451754385966,
        "commit_change": 0.8625017404556274,
        "descendant": 0.07775753736495972,
        "ell_ce": 5.236273864284158e-05,
        "emitted_length": 36.00137061403509,
        "entropy": 0.046696737408638,
        "extent_accuracy": 0.9999229311943054,
        "extent_exact": 0.9975328947368421,
        "extra": 0.001918859649122807,
        "h_mse": 0.07350583374500275,
        "length_accuracy": 0.9975328947368421,
        "log_prob": 0.0,
        "loss_per_example": 0.07804370671510696,
        "missing": 0.0005482456140350877,
        "missing_query": 0.0,
        "nonquery_accuracy": 0.9986726641654968,
        "nonquery_loss": 0.005122196860611439,
        "nonquery_nll": 0.004405738870324842,
        "nonquery_nll_support": 31.999451754385966,
        "partial_correct": 0.734375,
        "partial_support": 0.7368421052631579,
        "peer_reach": 0.0,
        "prediction_b_accuracy": 1.0,
        "prediction_extent_accuracy": 1.0,
        "prediction_extent_exact": 1.0,
        "query_accuracy": 0.03600146621465683,
        "query_exact": 0.0,
        "query_loss": 38.86627197265625,
        "query_nll": 38.86627143360021,
        "query_nll_support": 3.0,
        "structure_exact": 0.0,
        "true_length": 36.0,
        "valid": 1.0,
        "base_log_prob": -0.04698963463306427,
        "intervened": 1.0,
        "active_boundary_f1": 0.9075960859477454,
        "extent_class_support": [
          20640.0,
          224.0,
          224.0,
          224.0,
          7872.0
        ],
        "extent_class_recall": [
          1.0,
          0.9910714285714286,
          1.0,
          0.9988839285714286,
          1.0
        ],
        "partial_extent_accuracy": 0.9966517857142857
      },
      "paired": {
        "query_accuracy": {
          "delta": 0.9544956088066101,
          "ci95": [
            0.9468201994895935,
            0.9618055820465088
          ],
          "relative": 26.512685775756836,
          "relative_ci95": [
            22.370473861694336,
            32.59006118774414
          ],
          "support": 912,
          "replicates": 2000
        },
        "query_loss": {
          "delta": -38.82876205444336,
          "ci95": [
            -39.84673938751221,
            -37.75606079101563
          ],
          "relative": -0.9990348815917969,
          "relative_ci95": [
            -0.999446913599968,
            -0.9985292062163353
          ],
          "support": 912,
          "replicates": 2000
        },
        "query_nll": {
          "delta": -38.82875990177602,
          "ci95": [
            -39.88024442365629,
            -37.821971026660755
          ],
          "relative": -0.9990348564335976,
          "relative_ci95": [
            -0.9994605404520704,
            -0.9985138362284357
          ],
          "support": 912,
          "replicates": 2000
        }
      }
    }
  },
  "G3_eligible": true
}
```

See frozen G2/G3 protocols, selection records, per-seed/root/stratum evidence and validation ledger.
Engineering correctness, generated functionality, semantic-routing advantage and sampling-training advantage are separate claims.
No test-based selection, architecture/task redesign, H/I/Thermo changes or S4.
Phase H prerequisites: validated generated-structure interface/commit ownership, valid UNKNOWN encoding, repacking and shared-weight causal/global integration; no H science is authorized by this result.
