\version "2.10.33"

\header {
  title = "165 - O Lamm Gottes, unschuldig"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    f4 f4 c'4 c4 
    d2 c4 c4 
    f,4 g4 a4 bes4 
    a4 g8 f8 f4 } a4 
  a4 a4 g4 a4   %5
  f8 e8 d4 c4 c'4 
  d4 c4 c4 a8 bes8 
  c4 bes4 a4 d4 
  c8 bes8 a4 g4 a4 
  c4 a4 g2   %10
  f2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    c16 d16 ees8 ees8 d8 e8 c8 f8 e8 
    a4 g8 f8 e4 e4~ 
    e8 d4 c8 c8 d16 e16 f4~ 
    f8 e16 d16 e4 c4 } f4 
  f4 f4 f4 e4   %5
  d8 c4 b8 g4 ees'4 
  d8 e8 f8 g16 a16 g4 f4 
  a4. g4 fis8 g4 
  g4 f4 g4. f16 e16 
  f4 f2 e4   %10
  c2.
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  bes8 
    c4. bes16 a16 g4 c4~ 
    c4 b4 g4 a4 
    a4 g4 f4. d'8 
    c4 c8. bes16 a4 } c4 
  c4 d4 d4 a4   %5
  a8 g8 a8 g16 f16 e4 f4 
  f8 g8 a8 bes8 c4. d8 
  ees4 d4 d4 d8 f8 
  e4. d4 c8 c4 
  c8 a8 f8 a8 d4 c8 bes8   %10
  a2.
  
}

baixo = \relative c, {
  \repeat volta 2 {
    \partial 4 f8  g8 
    a4 bes2 a8 g8 
    f4 g4 c4 a4 
    d4 e4 f8 e8 d8 bes8 
    c4 c,4 f4 } f'4~ 
  f8 e8 d8 c8 b4 cis4   %5
  d8 e8 f8 g8 c,4 a4 
  bes4 f'4. e8 f4 
  f8 fis8 g8 g,8 d'4 bes8 b8 
  c4 d4 e4 f4 
  a,8 bes16 c16 d8 c8 bes4 c4   %10
  f,2. 
  
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