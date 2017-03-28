\version "2.10.33"

\header {
  title = "205 - Herr Gott, dich loben wir"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key a \minor
}

soprano = \relative c' {
  \partial 4 e4 
  g a c a
  a2 r4 c
  c b a a
  g2. \repeat volta 2 { c4
  c c b a  %% 05
  b a a a
  a g a b
  c a g } \repeat volta 2 { e
  g a b b
  a2. } b4  %% 10
  b b8 c d4 a
  a g a b
  c a g \repeat volta 2 { c4
  c c b a
  b a a a  %% 15
  a g a b
  c a g } \repeat volta 2 { a
  a a g b
  b a a a
  a g a8 b c4  %% 20
  g g e } f
  d f e d
  e fis g g
  g g a4. f8
  g4 f e c  %% 25
  d f e d
  f d c g'
  g g a4. f8
  g4 f e f
  f f e d  %% 30
  f d c g'
  g g a4. f8
  g4 f e e8 f
  g4 a a a
  c a a a  %% 35
  a g a8 b c4
  g g e \repeat volta 2 { a
  a a g b
  b b a a
  a g a c  %% 40
  g g e } f8 e
  d4 f g a
  g f e f
  f g a g8 f
  g4 f e g  %% 45
  e c d2
  e1~
  e  
}


alto = \relative c' {
  \partial 4 b4
  e f~ f8 e d4
  c2 r4 e8 fis
  g a~ a g~ g fis16 e fis4
  d2. \repeat volta 2 { g4~
  g8 fis16 e fis8 a~ a g16 fis e8 d  %% 05
  d g~ g fis16 e fis4 e8 g~
  g f~ f e16 d c4 d16 e f8
  e16 fis g8~ g fis d4 } \repeat volta 2 { c4
  d e8 d16 c f e d8~ d e16 d
  c2. } g'8 a  %% 10
  g4 g g~ g8 f16 e
  f4. e16 d c8 d16 e f4
  e16 fis g8~ g fis d4 \repeat volta 2 { g8 f
  e fis g a~ a g~ g fis
  g fis e4 fis d  %% 15
  c16 a d8~ d c~ c d16 c b8 d
  g4. fis8 d4 } \repeat volta 2 { f
  e16 f g8~ g fis~ fis e16 fis g4
  g8 f g4~ g8 f c f16 e
  f4. e8 f4 f~  %% 20
  f8 e d4 c} c~
  c8 b c d~ d c d4~
  d8 c16 b c8 b16 a b4 d
  e8 f g e c f f16 e d cis
  d8 a a d~ d cis a4  %% 25
  g c8 d~ d c~ c b
  a16 b c8~ c b g4 c
  c8 d e4~ e8 d16 cis d4~
  d8 e~ e d~ d cis d4~
  d8 c b d~ d c~ c b  %% 30
  c4. b8 g4 c
  c8 d e d16 e f4 f
  e4. d8 cis4 c
  b8 c16 d e8 d c d16 e f4
  e~ e16 d c b c4 e~  %% 35
  e8 f g4 f g8 f
  e4 d c \repeat volta 2 { e
  f e8 d16 c b4 b8 c16 e
  e4~ e16 fis g8~ g fis f4
  f c c c8 d  %% 40
  e4 e16 cis d8~ d cis} d4
  a8 bes16 c d4~ d8 e~ e d~
  d cis d4 cis d
  d~ d8 e f e d4
  d d16 cis d8 cis4 d  %% 45
  c g a gis8 a
  b c d4 c2~
  c4 b8 a b2

}


tenor = \relative c' {
  \partial 4 g4
  b d g,~ g8 f
  e2 r4 c'8 d
  e d d4 e16 d c4 d16 c
  b2. \repeat volta 2 {e4~
  e8 d16 c d4~ d c8 d16 c   %% 05
  b8 e16 d cis4 d e
  d4. c16 b a8 g f4
  g8 e' d8. c16 b4} \repeat volta 2 {g4~
  g8 a16 b c8 a~ a gis16 fis gis4
  e2.} e'8 dis   %% 10
  e4 e d e8 a,~
  a d16 c b8 c16 b a8 b16 c d4
  g,8 e' d8. c16 b4 \repeat volta 2 { c
  g~ g8 d' d4 d
  d4. cis8 d4 f,8 g   %% 15
  a f g4~ g8 fis d'4
  c8 e d8. c16 b4} \repeat volta 2 {d4
  a8 e'~ e d16 c b4 e
  d e d a8 bes
  c4 c c c~   %% 20
  c4. b8 g4} a
  g c,8 g' g4 g8 d
  a'4~ a8 d, d4 b'
  c8 d e c a c d16 e f8
  f e d a a4 e %% 25
  d a'8 g g4 g
  c,8 a' g8. f16 e4 e8 f
  g4 c8 bes a f~ f g16 a
  bes4 a a a
  g g g g %% 30
  f8 a g8. f16 e4 e8 f
  g a bes4 c d8. c16
  bes8 a a4 a a
  g8 a16 b c8 b a b16 c d4
  c16 b a g a8 f' e4 c8 d %% 35
  e d~ d c~ c d g, a
  b c4 b8 g4 \repeat volta 2 { c
  d a8 f'~ f e16 d e8 a,
  g fis g4 d' c~
  c8 bes bes4 a8 g f4   %% 40
  e8 g bes4 a} a8 g
  f4~ f8 g16 a bes8 c16 bes a8 f
  bes g a4 a a
  bes bes c bes
  bes a8 gis a4 g   %% 45
  g e f2
  e4 b'~ b a~
  a gis8 fis gis2
  
}


baixo = \relative c, {
  \partial 4 e4
  e' d c d
  a2 r4 a'
  e8 fis g4 c, d
  g,2. \repeat volta 2 { e'4
  a d,8 e16 fis g4. fis8    %% 05
  g e a4 d, cis
  d e f8 e d4
  c d g,} \repeat volta 2 { c
  b a8 f' d b e4
  a,2.} e'8 fis    %% 10
  g fis e4 b cis
  d g f8 e d4 
  c d g, \repeat volta 2 { e'8 d
  c d e fis g4 d
  g, a d, d'8 e    %% 15
  f4 e d g8 f
  e c d4 g,} \repeat volta 2 {d'
  cis d e4. fis8
  g4 cis, d f
  a,8 bes c4 f,8 g a b    %% 20
  c4 g c} f,
  g a8 b c4 b
  a d g, g'8 f
  e d c4 f8 g16 a bes4~
  bes8 cis, d4 a a  %% 25
  b a8 b c4 g
  f g c c8 d
  e4 c f, bes8 a
  g4 d' a d8 c
  b8 a g4 c g    %% 30
  a8 f g4 c, c'8 d
  e f g4 f8 e d4
  d8 cis d4 a a'
  e8 d c4 f8 e d4
  a'8 g f d a'4 a,8 b   %% 35
  c d e4. d8 e f
  g f g g, c4 \repeat volta 2 { a
  d c8 d e f g f
  e d cis4 d f8 e
  d4 e f a,8 b    %% 40
  c4 g a} d
  d8 c bes a g4 f
  e d a' d8 c
  bes a g4 f8 g16 a bes4~
  bes8 g d'4 a b    %% 45
  c2. b8 a
  gis4. e8 a b c d
  e2 e,


}


\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}