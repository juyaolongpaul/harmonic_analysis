\version "2.10.33"

\header {
  title = "309 - Ein Lämmlein geht und trägt die Schuld"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key aes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 ees4 
    f4 ees8 des8 c8 des8 ees4 
    des8 c8 des4 c4 bes4 
    c4 des4 ees4 des8 c8 
    bes8 aes8 bes4 aes4 } aes4 
  bes4 c4 des8 c8 bes4   %5
  c4 bes4 aes4 aes8 bes8 
  c4 des4 ees4 f4 
  c4 d4 ees4 c8 des8 
  ees4 f4 ees8 des8 c4 
  des2 c4 f4   %10
  f4 f4 bes,4 ees4 
  des4 c4 bes4 des4 
  c4 bes4 aes4 bes8 aes8 
  ges4 f4 ees4 ees4 
  aes4 bes4 c4. des8   %15
  ees4 des8 c8 bes4 c8 des8 
  bes2 aes4 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 aes4 
    aes4 g4 aes4 g4 
    f8 aes aes g aes4 bes
    aes4 aes4 aes4 aes4 
    aes4 g4 ees4 } ees4 
  ees8 f8 g4 f4 f8 ees8   %5
  ees4 des4 c4 f4 
  ees4 f4 ges4 f8 g8 
  aes8 g8 aes8 bes8 g4 ees4 
  aes4 aes4 bes4 c4 
  c8 a8 bes4 a4 c4   %10
  bes4 f8 des8 ees8 f8 ges4 
  f4 ees4 des4 bes'4 
  ees,8 f8 ges4 f8 ees8 f4
  f8 ees ees d ees4 ees8 des8 
  c8 bes8 ees4 ees aes   %15
  g4 aes2 aes4
  aes4 g4 ees4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    des8 c8 bes4 f'4 bes,8 c8 
    des8 f8 ees4 ees4 ees4 
    ees4 aes,8 bes8 c8 aes8 f'4 
    bes,8 c8 des4 c4 } c4 
  bes4 ees4 aes,4 bes4   %5
  bes8 aes aes g aes4 aes4 
  aes4 aes4 aes4 aes4 
  aes8 bes16 c16 bes4 bes4 aes4 
  aes8 ees'8 des8 c8 bes4 f'4 
  f2 f4 f8 ees8   %10
  des8 c8 bes4. a8 bes8 c8 
  a8 bes4 a8 bes4 f8 g8 
  aes4 ees'8 des8 ces4 bes4 
  bes4 aes4 g4 g4 
  aes4 aes8 g aes4 f'   %15
  ees2 f8 ees8 des4 
  ees8 f8 ees8 des8 c4 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 aes4 
    des,4 ees4 f4 g8 aes8 
    bes4 ees,4 aes4 g4 
    aes8 g8 f4 c4 des4 
    ees2 aes,4 } aes'4 
  g4 f8 ees8 f4 g4  %5
  aes4 ees4 aes,4 des4
  aes'8 ges8 f8 ees8 des8 c8 des8 ees8
  f4 bes,4 ees4 aes8 bes8
  c4 des4 g,4 a4
  bes4 bes,4 f'4 a,4  %10
  bes8 c8 des8 bes8 ges'4 f8 ees8
  f4 f,4 bes4 bes4
  c8 d8 ees4 ees d
  ees4 bes4 ees4 c4
  f4 ees4 aes8 g8 f4  %15
  c8 bes8 aes4 des8 c8 bes4
  ees8 des8 ees4 aes,4 
  
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